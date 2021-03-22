import os
import torch
#import torch.nn as nn
#import torch.optim as optim
#import matplotlib.pyplot as plt
#import torch.nn.functional as F
#from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/strus/all_PDFs.csv")

X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != 'true'], data.iloc[:,-1:], test_size=0.33, random_state=42)
print(X_train)
print(y_train)