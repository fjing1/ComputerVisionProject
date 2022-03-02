# Perform standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
# ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Define transforms for training and testing dataset
transforms = transforms.ToTensor()
# Load in our train and test set using dataloader objects
train_data = datasets.MNIST(root = '../data', train =True, download = True, transform = transforms)
# Display a batch of images
train_data = datasets.MNIST(root = '../data', train =False, download = True, transform = transforms)
train_loader =DataLoader(train_data, batch_size=10, shuffle=True)
test_loader =DataLoader(test_data, batch_size=10, shuffle=True)
# Define the model
conv1 = nn.Conv2d(1,6,3,1)
conv2 = nn.Conv2d(6,16, 3,1)
class CNN1(nn.Module):
    def __init__(self):
        self.conv1= nn.Conv2d(1,6,3,1)
        self.conv2= nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(2*2*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1,2*2*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.log_softmax(self.fc3(X), dim=1)
        return  X
# Instantiate the model, define loss and optimization functions
model = CNN1()
# Looking at the trainable parameters

# Train and test the model

# Save the trained model (so you can continue to train with more epochs later on,
# or you can grab those weight to perform evaluation on new images anytime you want)

# Evaluation using one test example
