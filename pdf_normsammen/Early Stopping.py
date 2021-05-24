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
import shutil
import scipy.stats
vej = 'pdf_notsammen'
#vej = 'pdf_normsammen'

shutil.rmtree('/home/nikolaj/Desktop/Bachelorprojekt/' + vej + '/train')
shutil.rmtree('/home/nikolaj/Desktop/Bachelorprojekt/' + vej + '/test')
os.makedirs('/home/nikolaj/Desktop/Bachelorprojekt/' + vej + '/train', exist_ok=True)
os.makedirs('/home/nikolaj/Desktop/Bachelorprojekt/' + vej + '/test', exist_ok=True)
new_path = '/home/nikolaj/Desktop/Bachelorprojekt/' + vej + '/train'
test_path = '/home/nikolaj/Desktop/Bachelorprojekt/' + vej + '/test'
data_PDF = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/" + vej + "/all.csv")
train_spilts = '.8'
test_val_spilts = '.9' #wight .9 for 10% test and 10% val .8 for 20%/20%

#data_PDF.hist(column= 'stru')

def pdf_train_test_val_splitter(data_PDF, train_spilts, test_val_spilts):
    data_PDF = data_PDF.sample(frac=1, random_state=1).reset_index(drop=True) #randomzie data
    stru_individuelt = dict()
    train_individuelt = dict()
    test_individuelt = dict()
    val_individuelt = dict()

    for k, v in data_PDF.groupby('stru'):
        stru_individuelt[k] = v
    stru_names = data_PDF['stru'].unique().tolist()
    train_all = []
    test_all = []
    val_all = []
    for i in stru_names: #makes a train, test and val dataframe for all individuelt structures
        train_spilt = float(train_spilts)
        test_val_spilt = float(test_val_spilts)
        data_PDF = stru_individuelt[i]
        train_spilt = int(train_spilt * len(data_PDF))  # 80%
        test_val_spilt = int(test_val_spilt * len(data_PDF))  # 10% for both
        train, validate, test = np.split(data_PDF.sample(frac=1,random_state=1), [train_spilt, test_val_spilt])
        train_all.append(train) #makes a list dataframes with all individuelt structures
        test_all.append(test)
        val_all.append(validate)

    train_merged = pd.concat(train_all)
    train_merged = train_merged.sample(frac=1, random_state=1).reset_index(drop=True) #dataframes now merged and randomized
    test_merged = pd.concat(test_all)
    test_merged = test_merged.sample(frac=1, random_state=1).reset_index(drop=True)
    val_merged = pd.concat(val_all)
    val_merged = val_merged.sample(frac=1, random_state=1).reset_index(drop=True)

    #dict so right dataframe can be taking - xxx_individuelt[stur]
    for k, v in train_merged.groupby('stru'):
        train_individuelt[k] = v
    for k, v in test_merged.groupby('stru'):
        test_individuelt[k] = v
    for k, v in val_merged.groupby('stru'):
        val_individuelt[k] = v
    return train_merged, test_merged, val_merged, train_individuelt, test_individuelt, val_individuelt

train_merged, test_merged, val_merged, train_individuelt, test_individuelt, val_individuelt = pdf_train_test_val_splitter(data_PDF,train_spilts,test_val_spilts)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path=new_path +'checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

size_train = train_merged["size"]
stru_train = train_merged["stru"]
size_test = test_merged["size"]
stru_test = test_merged["stru"]
size_val = val_merged["size"]
stru_val = val_merged["stru"]

train_merged = train_merged.drop(['stru'], axis=1)
train_merged = train_merged.drop(['size'], axis=1)
test_merged = test_merged.drop(['stru'], axis=1)
test_merged = test_merged.drop(['size'], axis=1)
val_merged = val_merged.drop(['stru'], axis=1)
val_merged = val_merged.drop(['size'], axis=1)

torch_tensor_train = torch.tensor(train_merged.values)
torch_tensor_test = torch.tensor(test_merged.values)
torch_tensor_val = torch.tensor(val_merged.values)

train_dataset = torch_tensor_train.float()
test_dataset = torch_tensor_test.float()
val_dataset = torch_tensor_val.float()

# print(len(train_dataset))size_spilt
# constants
n_epochs = 50000
LEARNING_RATE = 1e-3
batch_size = 200
patience = 500

size_loader = DataLoader(
    size_train,
    batch_size=batch_size,
    shuffle=False
)

stru_loader = DataLoader(
    stru_train,
    batch_size=batch_size,
    shuffle=False
)

trainloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False
)

# print(len(trainloader.dataset))
testloader = DataLoader(
    test_dataset,
    batch_size=len(test_merged),
    shuffle=False
)

valloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
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


def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
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


def save_dimi(dimi, epoch, size, stru, path):
    dimi = dimi.cpu().detach().numpy()
    norm = plt.Normalize(1, 4)
    c = np.array(size) / 100
    names = stru

    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    fig, ax = plt.subplots(figsize=(5, 8))
    sc = ax.scatter(dimi[:, 0], dimi[:, 1], c=c, cmap=cmap, s=80)
    plt.title("Latent Space for epoch{}".format(epoch))
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
    plt.savefig(path + '/nice_plot_{}.png'.format(epoch), dpi=200)
def save_dimi_test_indi(dimi, size, stru, stru_indi, path):
    dimi = dimi.cpu().detach().numpy()
    norm = plt.Normalize(1, 4)
    c = np.array(size) / 100
    names = stru
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    fig, ax = plt.subplots(figsize=(5, 8))
    sc = ax.scatter(dimi[:, 0], dimi[:, 1], c=c, cmap=cmap, s=80)
    plt.title("Latent Space for test_{}".format(stru_indi))
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
    plt.savefig(path + '/test_indi_{}.png'.format(stru_indi), dpi=200)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=301, out_features=150)
        self.enc2 = nn.Linear(in_features=150, out_features=80)
        self.enc3 = nn.Linear(in_features=80, out_features=40)
        self.enc4 = nn.Linear(in_features=40, out_features=20)
        self.enc5 = nn.Linear(in_features=20, out_features=10)
        self.enc6 = nn.Linear(in_features=10, out_features=5)
        self.enc7 = nn.Linear(in_features=5, out_features=2)

        self.dec1 = nn.Linear(in_features=2, out_features=5)
        self.dec2 = nn.Linear(in_features=5, out_features=10)
        self.dec3 = nn.Linear(in_features=10, out_features=20)
        self.dec4 = nn.Linear(in_features=20, out_features=40)
        self.dec5 = nn.Linear(in_features=40, out_features=80)
        self.dec6 = nn.Linear(in_features=80, out_features=120)
        self.dec7 = nn.Linear(in_features=120, out_features=150)
        self.dec8 = nn.Linear(in_features=150, out_features=301)
    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        x = torch.sigmoid(self.enc6(x))
        x = (self.enc7(x))

        z = x

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))
        x = torch.sigmoid(self.dec7(x))
        x = (self.dec8(x))
        return x, z


model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
latent_hver = 20


def train_model(model, trainloader, valloader, patience, n_epochs, size_loader, stru_loader):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):
        for size in size_loader:
            size = np.array(size)
        for stru in stru_loader:
            stru = np.array(stru)
        path = new_path
        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for batch, data in enumerate(trainloader, 1):
            # clear the gradients of all optimized variables
            target = data
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            dimi = output[1]
            output = output[0]
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        if epoch % latent_hver == 0:
            save_dimi(dimi, epoch, size, stru, path)
        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data in valloader:
            target = data
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            output = output[0]
            # calculate the loss
            loss = criterion(output, target)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{(epoch+1):>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(new_path + 'checkpoint.pt'))
    torch.save(model, new_path + 'full_model.pt')
    return model, avg_train_losses, avg_valid_losses, output

def test_PDF_reconstruction(model, testloader, size_test, stru_test):
    path = test_path
    for inte, data in enumerate(testloader):
        output = model(data)
        dimi = output[1]
        output = output[0]
        break
    size = np.array(size_test)
    stru = np.array(stru_test)
    save_dimi(dimi=dimi, epoch='test', size=size, stru=stru, path=path)
    return output

def test_PDF_reconstruction_indi(model, testloader, size_test, stru_test, stru_indi):
    path = test_path
    for inte, data in enumerate(testloader):
        output = model(data)
        dimi = output[1]
        output = output[0]
        break
    size = np.array(size_test)
    stru = np.array(stru_test)
    save_dimi_test_indi(dimi=dimi, size=size, stru=stru, path=path, stru_indi=stru_indi)
    return output


model, train_loss, valid_loss, train_output = train_model(model, trainloader, valloader, patience, n_epochs, size_loader, stru_loader)


# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1

max_y = (max(valid_loss)+0.1*max(valid_loss))
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, max_y)# consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig(new_path +'/loss_plot.png', bbox_inches='tight')

for i in range(5):
    train_output = train_output.data
    plt.figure()
    plt.plot(train_dataset[i], label='Original PDF')
    plt.plot(train_output[i], label='Decoded PDF')
    plt.plot((train_dataset[i]-train_output[i])- 0.75, label='Difference')
    a = scipy.stats.pearsonr(train_dataset[i], train_output[i])[0]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a))
    plt.xlabel('r [$\AA$]')
    plt.ylabel('G(r)')
    plt.legend()
    plt.savefig(new_path + '/sammelig_{}.png'.format(i))

test_output = test_PDF_reconstruction(model, testloader, size_test, stru_test)
for i in range(5):
    test_output = test_output.data
    plt.figure()
    plt.plot(test_dataset[i], label='Original PDF')
    plt.plot(test_output[i], label='Decoded PDF')
    plt.plot((test_dataset[i] - test_output[i]) - 0.75, label='Difference')
    a = scipy.stats.pearsonr(test_dataset[i], test_output[i])[0]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a))
    plt.xlabel('r [$\AA$]')
    plt.ylabel('G(r)')
    plt.legend()
    plt.savefig(test_path + '/sammelig_{}.png'.format(i))

stru_names = data_PDF['stru'].unique().tolist()

for indi_stru in stru_names:
    test_indi_stru = test_individuelt[indi_stru]
    size_indi_stru = test_indi_stru["size"]
    stru_indi_stru = test_indi_stru["stru"]
    test_indi_stru = test_indi_stru.drop(['stru'], axis=1)
    test_indi_stru = test_indi_stru.drop(['size'], axis=1)
    test_indi_stru = torch.tensor(test_indi_stru.values)
    test_indi_stru = test_indi_stru.float()
    testloader_indi_stru = DataLoader(
        test_indi_stru,
        batch_size=len(test_indi_stru),
        shuffle=False
    )
    test_PDF_recon = test_PDF_reconstruction_indi(model, testloader_indi_stru, size_indi_stru, stru_indi_stru, indi_stru)
    pre_indcode = test_PDF_recon[0]
    pre_ind = pre_indcode.data
    plt.figure()
    plt.plot(test_indi_stru[0], label='Original PDF')
    plt.plot(pre_ind, label='Decoded PDF')
    plt.plot((test_indi_stru[0] - pre_ind) - 0.75, label='Difference')
    a = scipy.stats.pearsonr(test_indi_stru[0], pre_ind)[0]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a))
    plt.title("Latent Space for test_{}".format(indi_stru))
    plt.xlabel('r [$\AA$]')
    plt.ylabel('G(r)')
    plt.legend()
    plt.savefig(test_path + '/sammelig_{}.png'.format(indi_stru))
