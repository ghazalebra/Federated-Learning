import torch
from torchvision import datasets, transforms
from collections import defaultdict
from torch.utils.data import Subset
import numpy as np
import os

import matplotlib.pyplot as plt

# set paths
DATA_DIR = './data'
SAVE_DIR = './splits'
os.makedirs(SAVE_DIR, exist_ok=True)

# set random seed for reproducibility
np.random.seed(42)

# download (if not present in the path) and load MNIST
transform = transforms.ToTensor()
mnist = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)

# group indices by label
class_indices = defaultdict(list)
for idx, (_, label) in enumerate(mnist):
    class_indices[label].append(idx)

# test without loading the dataset
# class_indices = {label: list(range(label * 10, (label + 1) * 10)) for label in range(10)}

# weights = {'1': [w1, w2, ..., wn], '2': [w1, w2, ..., wn], ...}
weights = {'1': [0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2], 
'2': [0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8]}

def skew_indices(weights, class_indices):
    skewed = defaultdict(list)
    for c in range(10):
        np.random.shuffle(class_indices[c])
        start = 0
        for client in weights.keys():
            end = start + int(weights[client][c]*len(class_indices[c]))
            skewed[client] += class_indices[c][start:end]
            start = end
    return skewed

def extract_skewed_data(weights, class_indices, save=True):

    # get skewed indices
    skewed_indices = skew_indices(weights, class_indices)

    # create subsets of MNIST
    clients_data = defaultdict(list)
    for client in skewed_indices.keys():
        subset = Subset(mnist, skewed_indices[client])
        
        if save:
            torch.save(subset, os.path.join(SAVE_DIR, f'client{client}.pt'))


extract_skewed_data(weights, class_indices, save=True)