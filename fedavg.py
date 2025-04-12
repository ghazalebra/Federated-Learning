# python fedavg.py --evaluate --model1 client1_model.pth --model2 client2_model.pth --global_model fedavg_model.pth
# python fedavg.py --model1 client1_model.pth --model2 client2_model.pth --output_model fedavg_model.pth

import torch
import argparse
from model import Classifier
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# paths
MODEL_DIR = './models'
SPLIT_DATA_DIR = './splits'
DATA_DIR = './data'

# standard fedavg algorithm
def standard_fedavg(path1, path2, output_path):
    # load client models paraneters
    state_dict1 = torch.load(MODEL_DIR + '/client1/' + path1)
    state_dict2 = torch.load(MODEL_DIR + '/client2/' + path2)

    # check if parameters match
    assert state_dict1.keys() == state_dict2.keys(), "Model parameters do not match."

    # take average (equal weights since same data size)
    averaged_state_dict = {}
    for key in state_dict1:
        averaged_state_dict[key] = 0.5 * state_dict1[key] + 0.5 * state_dict2[key]

    # save the aggregated model
    torch.save(averaged_state_dict, MODEL_DIR + '/' + output_path)
    print(f"FedAvg standard model saved to {output_path}")

# modified fedavg algorithm version 1: Averaging All Layers Except Classifier
def fedavg_v1(path1, path2, output_path):
    # load client models paraneters
    state_dict1 = torch.load(MODEL_DIR + '/client1/' + path1)
    state_dict2 = torch.load(MODEL_DIR + '/client2/' + path2)

    # check if parameters match
    assert state_dict1.keys() == state_dict2.keys(), "Model parameters do not match."

    # take average (equal weights since same data size)
    averaged_state_dict = {}
    for key in state_dict1:
        if "fc2" in key:
            # pick from better client (e.g. client1)
            averaged_state_dict[key] = state_dict1[key].clone()
        else:
            averaged_state_dict[key] = 0.5 * state_dict1[key] + 0.5 * state_dict2[key]

    # save the aggregated model
    torch.save(averaged_state_dict, MODEL_DIR + '/' + output_path)
    print(f"FedAvg v1 model saved to {output_path}")

# modified fedavg algorithm version 2: Class-wise Weighted FedAvg
def classwise_weighted_classifier(fc1, fc2, split_index=5, w_client1=0.8):
    # fc1, fc2 are nn.Linear layers from client1 and client2
    weight = torch.zeros_like(fc1.weight)
    bias = torch.zeros_like(fc1.bias)

    for i in range(weight.size(0)):  # 10 output classes
        if i < split_index:
            w = w_client1  # more trust in client 1
        else:
            w = 1 - w_client1  # more trust in client 2

        weight[i] = w * fc1.weight[i] + (1 - w) * fc2.weight[i]
        bias[i] = w * fc1.bias[i] + (1 - w) * fc2.bias[i]

    return weight, bias

def fedavg_v2(path1, path2, output_path):

    client1 = Classifier()
    client2 = Classifier()
    
    # load client models paraneters
    state_dict1 = torch.load(MODEL_DIR + '/client1/' + path1)
    state_dict2 = torch.load(MODEL_DIR + '/client2/' + path2)

    client1.load_state_dict(state_dict1)
    client2.load_state_dict(state_dict2)

    # check if parameters match
    assert state_dict1.keys() == state_dict2.keys(), "Model parameters do not match."

    # take average (equal weights since same data size)
    averaged_state_dict = {}
    for key in state_dict1:
        if "fc2.weight" in key or "fc2.bias" in key:
            continue 
        
        averaged_state_dict[key] = 0.5 * state_dict1[key] + 0.5 * state_dict2[key]

    # apply classwise weighting to classifier
    w, b = classwise_weighted_classifier(client1.fc2, client2.fc2)
    averaged_state_dict["fc2.weight"] = w
    averaged_state_dict["fc2.bias"] = b

    # save the aggregated model
    torch.save(averaged_state_dict, MODEL_DIR + '/' + output_path)
    print(f"FedAvg v2 model saved to {output_path}")

def evaluate(model, dataloader, device, desc="Evaluating"):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    # per-class accuracy
    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=desc, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    acc = correct / total
    avg_loss = total_loss / len(dataloader)

    per_class_acc = []
    for i in range(num_classes):
        acc_ = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        per_class_acc.append(acc_)

    return acc, avg_loss, per_class_acc

def main():
    parser = argparse.ArgumentParser(description="FedAvg")
    parser.add_argument("--evaluate", action='store_true', help="Evaluate")
    parser.add_argument("--model1", type=str, required=True, help="Path to first model")
    parser.add_argument("--model2", type=str, required=True, help="Path to second model")
    parser.add_argument("--output_model", type=str, required=False, help="Save Path for the global model")
    parser.add_argument("--global_model", type=str, required=False)
    parser.add_argument("--base_model", type=str, required=False, default='base_model.pth')

    # reproducibility
    torch.manual_seed(42)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # FedAvg
    if not args.evaluate:
        assert args.output_model, "Please provide the output path for the global model."
        out = args.output_model.replace(".pth", "")
        standard_fedavg(args.model1, args.model2, out + '0.pth')
        fedavg_v1(args.model1, args.model2, out + '1.pth')
        fedavg_v2(args.model1, args.model2, out + '2.pth')

    # evaluation
    else:
        
        # load models
        model1 = Classifier().to(device)
        model1.load_state_dict(torch.load(MODEL_DIR + '/client1/' + args.model1))

        model2 = Classifier().to(device)
        model2.load_state_dict(torch.load(MODEL_DIR + '/client2/' + args.model2))

        if args.base_model:
            base_model = Classifier().to(device)
            base_model.load_state_dict(torch.load(MODEL_DIR + '/' + args.base_model))
        
        global_model0 = Classifier().to(device)
        global_model1 = Classifier().to(device)
        global_model2 = Classifier().to(device)
        assert args.global_model, "Please provide the path to the global model."
        global_path = args.global_model.replace(".pth", "")
        try:
            global_model0.load_state_dict(torch.load(MODEL_DIR + '/' + global_path + '0.pth'))
            global_model1.load_state_dict(torch.load(MODEL_DIR + '/' + global_path + '1.pth'))
            global_model2.load_state_dict(torch.load(MODEL_DIR + '/' + global_path + '2.pth'))
        except:
            print('Global model not found! Running FedAvg function first...')
            out = args.global_model.replace(".pth", "")
            standard_fedavg(args.model1, args.model2, out + '0.pth')
            fedavg_v1(args.model1, args.model2, out + '1.pth')
            fedavg_v2(args.model1, args.model2, out + '2.pth')
            print('Done! Trying to evaluate again...')
            try:
                global_model0.load_state_dict(torch.load(MODEL_DIR + '/' + global_path + '0.pth'))
                global_model1.load_state_dict(torch.load(MODEL_DIR + '/' + global_path + '1.pth'))
                global_model2.load_state_dict(torch.load(MODEL_DIR + '/' + global_path + '2.pth'))
            except:
                print('Something went wrong!')
                return
        # load datasets
        client1_data = torch.load(SPLIT_DATA_DIR + '/' + 'client1.pt')
        client2_data = torch.load(SPLIT_DATA_DIR + '/' + 'client2.pt')
        transform = transforms.ToTensor()
        test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
        
        batch_size = 64
        client1_loader = DataLoader(client1_data, batch_size=batch_size, shuffle=False)
        client2_loader = DataLoader(client2_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # evaluate and display
        print("\n🔍 Evaluation Results:")
        acc1_own, loss1_own, per_class1_own = evaluate(model1, client1_loader, device, "Client 1 on its training split")
        acc2_own, loss2_own, per_class2_own = evaluate(model2, client2_loader, device, "Client 2 on its training split")
        acc1_other, loss1_other, per_class1_other = evaluate(model1, client2_loader, device, "Client 1 on the other training split")
        acc2_other, loss2_other, per_class2_other = evaluate(model2, client1_loader, device, "Client 2 on the other training split")

        acc1_test, _, per_class1 = evaluate(model1, test_loader, device, "Client 1 on test set")
        acc2_test, _, per_class2 = evaluate(model2, test_loader, device, "Client 2 on test set")

        acc_global0, loss_global0, per_class_global0 = evaluate(global_model0, test_loader, device, "Global model 0 on test set")
        acc_global1, loss_global1, per_class_global1 = evaluate(global_model1, test_loader, device, "Global model 1 on test set")
        acc_global2, loss_global2, per_class_global2 = evaluate(global_model2, test_loader, device, "Global model 2 on test set")
        
        acc_base, _, per_class_base = evaluate(base_model, test_loader, device, "base model on test set")

        print(f"\n📊 Accuracy on own training split:")
        print(f"Client 1 per-class: {', '.join([f'{x:.4f}' for x in per_class1_own])}, Total: {acc1_own:.4f}")
        print(f"Client 2 per-class: {', '.join([f'{x:.4f}' for x in per_class2_own])}, Total: {acc2_own:.4f}")

        print(f"\n📊 Accuracy on the other training split:")
        print(f"Client 1 per-class: {', '.join([f'{x:.4f}' for x in per_class1_other])}, Total: {acc1_other:.4f}")
        print(f"Client 2 per-class: {', '.join([f'{x:.4f}' for x in per_class2_other])}, Total: {acc2_other:.4f}")

        print(f"\n📊 Accuracy on shared test set:")
        # print(f"Base Model per-class: {', '.join([f'{x:.4f}' for x in per_class_base])}, Total: {acc_base:.4f}")
        print(f"Client 1 per-class: {', '.join([f'{x:.4f}' for x in per_class1])}, Total: {acc1_test:.4f}")
        print(f"Client 2 per-class: {', '.join([f'{x:.4f}' for x in per_class2])}, Total: {acc2_test:.4f}")

        print(f"FedAvg 0 per-class: {', '.join([f'{x:.4f}' for x in per_class_global0])}, Total: {acc_global0:.4f}")
        print(f"FedAvg 1 per-class: {', '.join([f'{x:.4f}' for x in per_class_global1])}, Total: {acc_global1:.4f}")
        print(f"FedAvg 2 per-class: {', '.join([f'{x:.4f}' for x in per_class_global2])}, Total: {acc_global2:.4f}")



if __name__ == "__main__":
    main()
