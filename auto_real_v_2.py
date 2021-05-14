# import packages
import os
import sys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
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
import torch.optim as optim
from torch.distributions import Normal, Independent
from ase.data import covalent_radii, atomic_numbers, chemical_symbols
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import umap
import scipy.stats

os.makedirs('/home/nikolaj/Desktop/Bachelorprojekt/pdf_normsammen/train', exist_ok=True)
os.makedirs('/home/nikolaj/Desktop/Bachelorprojekt/pdf_normsammen/test', exist_ok=True)
new_path = 'pdf_normsammen/train'
test_path = 'pdf_normsammen/test'
data_PDF = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/pdf_normsammen/all.csv")
data_PDF = data_PDF.sample(frac=1).reset_index(drop=True)
size = data_PDF["size"]
stru = data_PDF["stru"]
data_PDF = data_PDF.drop(['stru'], axis=1)
data_PDF = data_PDF.drop(['size'], axis=1)
torch_tensor = torch.tensor(data_PDF.values)
DATASET_SIZE = torch_tensor.size(0)

train_size = int(0.8 * DATASET_SIZE)
val_size = int(0.1*DATASET_SIZE)
test_size = int(0.1*DATASET_SIZE)

data = torch.split(torch_tensor, [train_size, test_size, val_size])
train_dataset = data[0].float()
test_dataset = data[1].float()
val_dataset = data[2].float()

size_train = size.iloc(axis=0)[:train_size]
size_test = size_train.iloc(axis=0)[:test_size]
size_val = size_test.iloc(axis=0)[:val_size]

stru_train = stru.iloc(axis=0)[:train_size]
stru_test = stru_train.iloc(axis=0)[:test_size]
stru_val = stru_test.iloc(axis=0)[:val_size]

#print(len(train_dataset))size_spilt
# constants
NUM_EPOCHS = 1221
LEARNING_RATE = 1e-3
BATCH_SIZE = 242

size_loader = DataLoader(
    size_train,
    batch_size=BATCH_SIZE,
    shuffle=False
)

stru_loader = DataLoader(
    stru_train,
    batch_size=BATCH_SIZE,
    shuffle=False
)

trainset = train_dataset
testset = test_dataset
valset = val_dataset

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

#print(len(trainloader.dataset))
testloader = DataLoader(
    testset,
    batch_size=test_size,
    shuffle=False
)

valloader = DataLoader(
    valset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


data_loaders = {"train": trainloader, "val": valloader}
data_lengths = {"train": len(trainloader.dataset), "val": len(valloader.dataset)}

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

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))),
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()
                
def save_dimi(dimi, epoch, size, stru):
    dimi = dimi.cpu().detach().numpy()
    norm = plt.Normalize(1, 4)
    c = np.array(size) / 100
    names = stru

    #for i in range(len(stru)):
        #names.append(stru[i][:stru[i].find("_")] + "-" + str(
            #int(stru.tolist()[i])))
 
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    fig, ax = plt.subplots(figsize=(5, 8))
    sc = ax.scatter(dimi[:,0], dimi[:,1], c=c, cmap=cmap, s=80)
    plt.title("Latent Space for epoch{}".format(epoch))
    plt.xlabel("Latent Space Variable 1")
    plt.ylabel("Latent Space Variable 2")
    plt.yticks([])
    plt.xticks([])
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False)
    #plt.legend()
    plt.tight_layout()

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    # loop through each x,y pair
    for iter, (i, j) in enumerate(zip(dimi[:,0], dimi[:,1])):
        ax.annotate(stru[iter][:3], xy=(i, j), color='black',
                    fontsize="x-small", weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')

    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.savefig("./" + new_path + '/nice_plot_{}.png'.format(epoch), dpi=200)


def save_dimi_test(dimi, size, stru):
    dimi = dimi.cpu().detach().numpy()
    norm = plt.Normalize(1, 4)
    c = np.array(size) / 100
    names = stru

    # for i in range(len(stru)):
    # names.append(stru[i][:stru[i].find("_")] + "-" + str(
    # int(stru.tolist()[i])))

    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    fig, ax = plt.subplots(figsize=(5, 8))
    sc = ax.scatter(dimi[:, 0], dimi[:, 1], c=c, cmap=cmap, s=80)
    plt.title("Latent Space for test")
    plt.xlabel("Latent Space Variable 1")
    plt.ylabel("Latent Space Variable 2")
    plt.yticks([])
    plt.xticks([])
    right_side = ax.spines["right"]
    right_side.set_visible(False)
    top = ax.spines["top"]
    top.set_visible(False)
    # plt.legend()
    plt.tight_layout()

    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    # loop through each x,y pair
    for iter, (i, j) in enumerate(zip(dimi[:, 0], dimi[:, 1])):
        ax.annotate(stru[iter][:3], xy=(i, j), color='black',
                    fontsize="x-small", weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')

    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.savefig("./" + test_path + '/nice_plot_test.png', dpi=600)

#encoder
class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=301, out_features=150)
        self.enc2 = nn.Linear(in_features=150, out_features=80)
        self.enc3 = nn.Linear(in_features=80, out_features=40)
        self.enc4 = nn.Linear(in_features=40, out_features=20)
        self.enc5 = nn.Linear(in_features=20, out_features=10)
        self.enc6 = nn.Linear(in_features=10, out_features=5)
        self.enc7 = nn.Linear(in_features=5, out_features=2)
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = torch.sigmoid(self.enc6(x))
        x = (self.enc7(x))
        return x

#decoder
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.dec1 = nn.Linear(in_features=2, out_features=5)
        self.dec2 = nn.Linear(in_features=5, out_features=10)
        self.dec3 = nn.Linear(in_features=10, out_features=20)
        self.dec4 = nn.Linear(in_features=20, out_features=40)
        self.dec5 = nn.Linear(in_features=40, out_features=80)
        self.dec6 = nn.Linear(in_features=80, out_features=120)
        self.dec7 = nn.Linear(in_features=120, out_features=150)
        self.dec8 = nn.Linear(in_features=150, out_features=301)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))
        x = torch.sigmoid(self.dec7(x))
        x = (self.dec8(x))
        return x

end = encoder()
de = decoder()

criterion = nn.MSELoss()
#optimizer_end = optim.Adam(end.parameters(), lr=LEARNING_RATE)

optimizer = optim.Adam([
                {'params': end.parameters()},
                {'params': de.parameters()}
            ], lr=LEARNING_RATE)

def train(end, de, data_loaders, data_lengths, NUM_EPOCHS=NUM_EPOCHS, size_loader=size_loader, stru_loader=stru_loader):
    train_loss = []
    val_loss = []
    for epoch in range(NUM_EPOCHS):
        for phase in ['train', 'val']:
            if phase == 'train':
                end.train(True)  # Set model to training mode
                de.train(True)
            else:
                end.train(False)  # Set model to evaluate mode
                de.train(False)
            running_loss = 0.0
            for data in data_loaders[phase]:
                img = data
                img = img.to(device)
                optimizer.zero_grad()
                outputs = end(img)
                dimi = outputs
                outputs = de(outputs)
                loss = criterion(outputs, img)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
            for size in size_loader:
                size = np.array(size)
            for stru in stru_loader:
                stru = np.array(stru)
            running_loss += loss.item()
            epoch_loss = running_loss / data_lengths[phase]
            if phase == 'train':
                train_loss.append(epoch_loss)
            if phase == 'val':
                val_loss.append(epoch_loss)
            print('Epoch {} of {}, Train Loss: {:.3f}'.format(
                epoch + 1, NUM_EPOCHS, epoch_loss))
            if phase == 'train':
                if epoch % 20 == 0:
                    save_dimi(dimi.cpu().data, epoch, size, stru)
    return train_loss, val_loss, outputs


def test_PDF_reconstruction(end, de, testloader, size_test, stru_test):
    for inte, batch in enumerate(testloader):
        img = batch
        img = img.to(device)
        outputs = end(img)
        dimi = outputs
        outputs = de(outputs)
        break
    size = np.array(size_test)
    stru = np.array(stru_test)
    #print(len(size_test))
    save_dimi_test(dimi.cpu().data, size, stru)
    return outputs

# get the computation device
device = get_device()
print(device)
# load the neural network onto the device

end.to(device)
de.to(device)
make_dir()
# train the network
train_loss = train(end, de, data_loaders, data_lengths, NUM_EPOCHS, size_loader, stru_loader)
plt.figure()
plt.plot(train_loss[0], label='train_loss')
plt.plot(train_loss[1], label='val_loss')
plt.legend()
plt.title('Train/val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig("./" + new_path +'/deep_ae_loss.png')
# test the network
#test_image_reconstruction(de, testloader)
#print(train_loss[1])

#print(len(train_loss[2]))
#print(train_loss[2])
for i in range(5):
    pre_indcode = train_loss[2]
    pre_ind = pre_indcode.data
    plt.figure()
    plt.plot(trainset[i], label='original PDF')
    plt.plot(pre_ind[i], label='Decoded PDF')
    a = scipy.stats.pearsonr(trainset[i], pre_ind[i])[0]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a))
    plt.xlabel('r [$\AA$]')
    plt.ylabel('G(r)')
    plt.legend()
    plt.savefig("./" + new_path + '/sammelig_{}.png'.format(i))
print(train_loss[2])
test_PDF_recon = test_PDF_reconstruction(end, de, testloader, size_test, stru_test)
for i in range(5):
    pre_indcode = test_PDF_recon[i]
    pre_ind = pre_indcode.data
    plt.figure()
    plt.plot(testset[i], label='original PDF')
    plt.plot(pre_ind,label='Decoded PDF')
    a = scipy.stats.pearsonr(testset[i], pre_ind)[0]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a))
    plt.xlabel('r [$\AA$]')
    plt.ylabel('G(r)')
    plt.legend()
    plt.savefig("./" + test_path +'/sammelig_{}.png'.format(i))

print(test_PDF_recon)



