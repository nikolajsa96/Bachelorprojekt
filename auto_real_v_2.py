# import packages
import os
import sys

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

new_path = 'Auto_real_data_dec_oct'
data_PDF = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/Sigsig_data/dec_oct.csv")

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


#print(len(train_dataset))
# constants
NUM_EPOCHS = 11
LEARNING_RATE = 1e-3
BATCH_SIZE = 16

# image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = train_dataset
testset = test_dataset
print(len(trainset))
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

#print(len(trainloader.dataset))
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
    pdf_dir = new_path
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
def save_dimi(dimi, epoch):
    plt.figure()
    dimi = dimi.cpu().detach().numpy()
    dimi_len =len(dimi)
    df = pd.DataFrame(dimi, columns=['den_x', 'den_y'])
    df.plot.scatter(x='den_x', y='den_y')
    #plt.xlim(np.min(df['den_x']), np.max(df['den_x']))
    #plt.ylim(np.min(df['den_y']), np.max(df['den_y']))
    plt.title(dimi_len)
    plt.savefig("./" + new_path + '/dimi_image{}.png'.format(epoch))

#encoder

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=3001, out_features=1500)
        self.enc2 = nn.Linear(in_features=1500, out_features=800)
        self.enc3 = nn.Linear(in_features=800, out_features=400)
        self.enc4 = nn.Linear(in_features=400, out_features=200)
        self.enc5 = nn.Linear(in_features=200, out_features=100)
        self.enc6 = nn.Linear(in_features=100, out_features=50)
        self.enc7 = nn.Linear(in_features=50, out_features=20)
        self.enc8 = nn.Linear(in_features=20, out_features=10)
        self.enc9 = nn.Linear(in_features=10, out_features=5)
        self.enc10 = nn.Linear(in_features=5, out_features=2)
    def forward(self, x):
        x = (self.enc1(x))
        x = (self.enc2(x))
        x = (self.enc3(x))
        x = (self.enc4(x))
        x = (self.enc5(x))
        x = (self.enc6(x))
        x = (self.enc7(x))
        x = (self.enc8(x))
        x = (self.enc9(x))
        x = (self.enc10(x))
        return x

#decoder
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.dec1 = nn.Linear(in_features=2, out_features=5)
        self.dec2 = nn.Linear(in_features=5, out_features=10)
        self.dec3 = nn.Linear(in_features=10, out_features=20)
        self.dec4 = nn.Linear(in_features=20, out_features=50)
        self.dec5 = nn.Linear(in_features=50, out_features=100)
        self.dec6 = nn.Linear(in_features=100, out_features=200)
        self.dec7 = nn.Linear(in_features=200, out_features=400)
        self.dec8 = nn.Linear(in_features=400, out_features=800)
        self.dec9 = nn.Linear(in_features=800, out_features=1200)
        self.dec10 = nn.Linear(in_features=1200, out_features=1500)
        self.dec11 = nn.Linear(in_features=1500, out_features=3001)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = (self.dec2(x))
        x = (self.dec3(x))
        x = F.relu(self.dec4(x))
        x = (self.dec5(x))
        x = (self.dec6(x))
        x = torch.sigmoid(self.dec7(x))
        x = (self.dec8(x))
        x = (self.dec9(x))
        x = torch.sigmoid(self.dec10(x))
        x = (self.dec11(x))
        return x

end = encoder()
de = decoder()

criterion = nn.MSELoss()
optimizer_end = optim.Adam(end.parameters(), lr=LEARNING_RATE)
#optimizer_de = optim.Adam(de.parameters(), lr=LEARNING_RATE)

def train(end, de, trainloader, NUM_EPOCHS=NUM_EPOCHS):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for iter, data in enumerate(trainloader):
            print(data.size())
            print(iter)
            img = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            print(img.size())
            optimizer_end.zero_grad()
            #optimizer_de.zero_grad()
            outputs = end(img)
            dimi = outputs
            outputs = de(outputs)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer_end.step()
            #optimizer_de.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, NUM_EPOCHS, loss))
        if epoch % 5 == 0:
            save_dimi(dimi.cpu().data, epoch)
    return train_loss, dimi, outputs


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

end.to(device)
de.to(device)
make_dir()
# train the network
train_loss = train(end, de, trainloader, NUM_EPOCHS)
plt.figure()
plt.plot(train_loss[0])
plt.title('Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig("./" + new_path +'/deep_ae_loss.png')
# test the network
#test_image_reconstruction(de, testloader)
#print(train_loss[1])

print(len(train_loss[2]))
print(train_loss[2])
pre_indcode = train_loss[2]
pre_ind = pre_indcode.data
plt.figure()
plt.plot(testset[0])
plt.plot(pre_ind[0])
plt.savefig("./" + new_path +'/sammelig.png')
