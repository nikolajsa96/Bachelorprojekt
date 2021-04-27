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

# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
def make_dir():
    image_dir = 'FashionMNIST_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
def save_decoded_image(img, epoch):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, './FashionMNIST_Images/linear_ae_image{}.png'.format(epoch))

#encoder

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=784, out_features=256)
        self.enc2 = nn.Linear(in_features=256, out_features=128)
        self.enc3 = nn.Linear(in_features=128, out_features=64)
        self.enc4 = nn.Linear(in_features=64, out_features=32)
        self.enc5 = nn.Linear(in_features=32, out_features=16)
        self.enc6 = nn.Linear(in_features=16, out_features=8)
        self.enc7 = nn.Linear(in_features=8, out_features=4)
        self.enc8 = nn.Linear(in_features=4, out_features=2)
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = F.relu(self.enc6(x))
        x = F.relu(self.enc7(x))
        x = F.relu(self.enc8(x))
        return x

#decoder
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.dec1 = nn.Linear(in_features=2, out_features=4)
        self.dec2 = nn.Linear(in_features=4, out_features=8)
        self.dec3 = nn.Linear(in_features=8, out_features=16)
        self.dec4 = nn.Linear(in_features=16, out_features=32)
        self.dec5 = nn.Linear(in_features=32, out_features=64)
        self.dec6 = nn.Linear(in_features=64, out_features=128)
        self.dec7 = nn.Linear(in_features=128, out_features=256)
        self.dec8 = nn.Linear(in_features=256, out_features=784)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))
        x = F.relu(self.dec7(x))
        x = F.relu(self.dec8(x))
        return x

end = encoder()
de = decoder()



criterion = nn.MSELoss()
optimizer = optim.Adam(end.parameters(), lr=LEARNING_RATE)


def train(end, de, trainloader, NUM_EPOCHS=NUM_EPOCHS):
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = end(img)
            outputs = de(outputs)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, NUM_EPOCHS, loss))
        if epoch % 5 == 0:
            save_decoded_image(outputs.cpu().data, epoch)
    return train_loss


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
plt.savefig('deep_ae_loss.png')
# test the network
#test_image_reconstruction(de, testloader)
#print(train_loss[1])

