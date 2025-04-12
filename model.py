import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3) # 1 × 28 × 28 -> 16 x 28 x 28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # -> 16 x 13 x 13
        self.fc1 = nn.Linear(2704, 128) # 2704 -> 128
        self.fc2 = nn.Linear(128, 10)  # -> 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # output logits
        return x