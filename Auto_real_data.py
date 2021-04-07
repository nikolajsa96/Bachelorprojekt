# import packages
import os
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import pandas as pd
from sklearn.model_selection import train_test_split
#import tensorflow as tf

data_PDF = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/strus/all_PDFs.csv")
data_PDF = data_PDF.drop(['true'], axis=1)
data_PDF = data_PDF.sample(frac=1).reset_index(drop=True)
torch_tensor = torch.tensor(data_PDF.values)
DATASET_SIZE = torch_tensor.size(0)

train_size = int(0.7 * DATASET_SIZE)
#val_size = int(0.15 * DATASET_SIZE)
test_size = int(DATASET_SIZE-train_size)

data = torch.split(torch_tensor, [train_size, test_size])
train_dataset = data[0].float()
test_dataset = data[1].float()

# constants
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

# image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = train_dataset
testset = test_dataset
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
def make_dir():
    pdf_dir = 'Auto_real_data'
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
      
def save_decoded_pdf(outputs, epoch):
    #img = img.view(img.size(0), img.size(1))
    plt.figure()
    plt.plot(outputs)
    plt.savefig('./FashionMNIST_Images/linear_ae_image{}.png'.format(epoch))


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=300, out_features=250)
        self.enc2 = nn.Linear(in_features=250, out_features=200)
        self.enc3 = nn.Linear(in_features=200, out_features=180)
        self.enc4 = nn.Linear(in_features=180, out_features=150)
        self.enc5 = nn.Linear(in_features=150, out_features=130)
        self.enc6 = nn.Linear(in_features=130, out_features=100)
        self.enc7 = nn.Linear(in_features=100, out_features=80)
        self.enc8 = nn.Linear(in_features=80, out_features=50)
        self.enc9 = nn.Linear(in_features=50, out_features=30)
        self.enc10 = nn.Linear(in_features=30, out_features=15)
        #self.enc11 = nn.Linear(in_features=15, out_features=10)
        #self.enc12 = nn.Linear(in_features=10, out_features=4)
        #self.enc13 = nn.Linear(in_features=4, out_features=2)
        # decoder
        #self.dec1 = nn.Linear(in_features=2, out_features=4)
        #self.dec2 = nn.Linear(in_features=4, out_features=10)
        #self.dec3 = nn.Linear(in_features=10, out_features=15)
        self.dec4 = nn.Linear(in_features=15, out_features=30)
        self.dec5 = nn.Linear(in_features=30, out_features=50)
        self.dec6 = nn.Linear(in_features=50, out_features=80)
        self.dec7 = nn.Linear(in_features=80, out_features=100)
        self.dec8 = nn.Linear(in_features=100, out_features=130)
        self.dec9 = nn.Linear(in_features=130, out_features=150)
        self.dec10 = nn.Linear(in_features=150, out_features=180)
        self.dec11 = nn.Linear(in_features=180, out_features=200)
        self.dec12 = nn.Linear(in_features=200, out_features=250)
        self.dec13 = nn.Linear(in_features=250, out_features=300)
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc6(x))
        x = F.relu(self.enc7(x))
        x = F.relu(self.enc8(x))
        x = F.relu(self.enc9(x))
        x = F.relu(self.enc10(x))
        #x = F.relu(self.enc11(x))
        #x = F.relu(self.enc12(x))
        #x = F.relu(self.enc13(x))
        z = x.clone()
        #x = F.relu(self.dec1(x))
        #x = F.relu(self.dec2(x))
        #x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))
        x = F.relu(self.dec7(x))
        x = F.relu(self.dec8(x))
        x = F.relu(self.dec9(x))
        x = F.relu(self.dec10(x))
        x = F.relu(self.dec11(x))
        x = F.relu(self.dec12(x))
        x = F.relu(self.dec13(x))
        return x, z
net = Autoencoder()
print(net)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


def train(net, trainloader, NUM_EPOCHS):
    train_loss = []
    dimi_list = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            img = data
            img = img.to(device)
            #img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = net(img)
            dimi = outputs[1]
            outputs = outputs[0]
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        dimi_list.append(dimi)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, NUM_EPOCHS, loss))
        #if epoch % 5 == 0:
            #save_decoded_pdf(outputs.cpu().data, epoch)
    return train_loss, dimi_list, outputs


def test_image_reconstruction(net, testloader):
    for batch in testloader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs = net(img)
        outputs = outputs[0]
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        save_image(outputs, 'fashionmnist_reconstruction.png')
        break

# get the computation device
device = get_device()
print(device)
# load the neural network onto the device
net.to(device)
make_dir()
# train the network
train_loss = train(net, trainloader, NUM_EPOCHS)
plt.figure()
plt.plot(train_loss[0])
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('deep_ae_fashionmnist_loss.png')
# test the network
#test_image_reconstruction(net, testloader)
#print(train_loss[1].shape())
pre_indcode = train_loss[2]

pre_ind = pre_indcode.data
plt.figure()
plt.plot(testset[0])
plt.plot(pre_ind[0])

plt.show()


