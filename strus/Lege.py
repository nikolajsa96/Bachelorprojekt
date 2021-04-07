import os
import torch
#import torch.nn as nn
#import torch.optim as optim
#import matplotlib.pyplot as plt
#import torch.nn.functional as F
#from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
"""
data = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/strus/all_PDFs.csv")

X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != 'true'], data.iloc[:,-1:], test_size=0.33, random_state=42)
print(X_train)
print(y_train)
"""
# import packages
import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image




# constants
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
# image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
testset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

for data in trainloader:
    print(data)
