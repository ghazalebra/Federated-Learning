# python client_training.py --client_id 1 --output_model client1_model.pth

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from model import Classifier

def train(model, dataloader, device, epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_list = []
    acc_list = []

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = correct / total

        loss_list.append(epoch_loss)
        acc_list.append(epoch_acc)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    return loss_list, acc_list


def plot_metrics(loss, acc, output_prefix):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss)
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(acc)
    plt.title("Accuracy")
    plt.savefig(f"{output_prefix}_metrics.png")
    plt.close()


def main():

    # reproducibility
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=False)
    parser.add_argument("--output_model", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    if args.client_id:
        client_data = torch.load(f'./splits/client{args.client_id}.pt', weights_only=False)
    else:
        transform = transforms.ToTensor()
        client_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(client_data, batch_size=64, shuffle=True)

    # initialize and train model
    model = Classifier().to(device)
    loss_list, acc_list = train(model, dataloader, device)

    # save model and metrics
    if args.client_id:
        save_dir = f'./models/client{args.client_id}'
    else:
        save_dir = f'./models/'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_dir + '/' + args.output_model)
    torch.save({'loss': loss_list, 'accuracy': acc_list}, save_dir + f'/{args.output_model.replace(".pth", "")}_metrics.pt')
    plot_metrics(loss_list, acc_list, args.output_model.replace(".pth", ""))


if __name__ == "__main__":
    main()
