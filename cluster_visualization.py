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
from scipy.spatial import ConvexHull
import shutil
import mendeleev
import scipy.stats

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

vej = 'pdf_notsammen'
#vej = 'pdf_normsammen'
model = torch.load('/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/trainfull_model.pt')
model.eval()
new_path = '/home/nikolaj/Desktop/Bachelorprojekt/' + vej + '/train'
test_path = '/home/nikolaj/Desktop/Bachelorprojekt/' + vej + '/test'
data_PDF = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/" + vej + "/all.csv")
train_spilts = '.8'
test_val_spilts = '.9' #wight .9 for 10% test and 10% val .8 for 20%/20%

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


def save_dimi_test_indi(test_individuelt, stru, size, stru_indi, path):
    norm = plt.Normalize(1, 4)
    c = np.array(size) / 100
    names = stru
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    fig, ax = plt.subplots(figsize=(5, 8))
    sc = ax.scatter(test_individuelt['Latent_Space_Variable_1'], test_individuelt['Latent_Space_Variable_2'], c=c, cmap=cmap, s=80)
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
    for iter, (i, j) in enumerate(zip(test_individuelt['Latent_Space_Variable_1'], test_individuelt['Latent_Space_Variable_2'])):
        ax.annotate(stru[iter], xy=(i, j), color='black',
                    fontsize="x-small", weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')

    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.savefig(path + '/person_plot_{}.png'.format(stru_indi), dpi=200)

def pdf_train_test_val_splitter(data_PDF, train_spilts, test_val_spilts):
    data_PDF = data_PDF.sample(frac=1).reset_index(drop=True) #randomzie data
    stru_individuelt = dict()
    train_individuelt = dict()
    test_individuelt = dict()
    val_individuelt = dict()
    first_peak_data = pd.DataFrame(data_PDF.iloc[:, :300].idxmax(axis=1), columns={'first_peak'}, dtype='int32')
    data_PDF = pd.concat([data_PDF, first_peak_data], axis=1, join='inner')
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
        train, validate, test = np.split(data_PDF.sample(frac=1), [train_spilt, test_val_spilt])
        train_all.append(train) #makes a list dataframes with all individuelt structures
        test_all.append(test)
        val_all.append(validate)

    train_merged = pd.concat(train_all)
    train_merged = train_merged.sample(frac=1).reset_index(drop=True) #dataframes now merged and randomized
    test_merged = pd.concat(test_all)
    test_merged = test_merged.sample(frac=1).reset_index(drop=True)
    val_merged = pd.concat(val_all)
    val_merged = val_merged.sample(frac=1).reset_index(drop=True)

    #dict so right dataframe can be taking - xxx_individuelt[stur]
    for k, v in train_merged.groupby('stru'):
        train_individuelt[k] = v
    for k, v in test_merged.groupby('stru'):
        test_individuelt[k] = v
    for k, v in val_merged.groupby('stru'):
        val_individuelt[k] = v
    return train_merged, test_merged, val_merged, train_individuelt, test_individuelt, val_individuelt

train_merged, test_merged, val_merged, train_individuelt, test_individuelt, val_individuelt = pdf_train_test_val_splitter(data_PDF,train_spilts,test_val_spilts)

def test_PDF_reconstruction(model, testloader, size_test, stru_test):
    for inte, data in enumerate(testloader):
        output = model(data)
        dimi = output[1]
        output = output[0]
        break
    size = np.array(size_test)
    stru = np.array(stru_test)

    return output, dimi, size, stru, data
"""
size_train = train_merged["size"]
stru_train = train_merged["stru"]
size_test = test_merged["size"]
stru_test = test_merged["stru"]
size_val = val_merged["size"]
stru_val = val_merged["stru"]
atom_train = train_merged["atom"]
atom_test = test_merged["atom"]
atom_val = val_merged["atom"]

train_merged = train_merged.drop(['stru'], axis=1)
train_merged = train_merged.drop(['size'], axis=1)
train_merged = train_merged.drop(['atom'], axis=1)
test_merged = test_merged.drop(['stru'], axis=1)
test_merged = test_merged.drop(['size'], axis=1)
test_merged = test_merged.drop(['atom'], axis=1)
val_merged = val_merged.drop(['stru'], axis=1)
val_merged = val_merged.drop(['size'], axis=1)
val_merged = val_merged.drop(['atom'], axis=1)

torch_tensor_train = torch.tensor(train_merged.values)
torch_tensor_test = torch.tensor(test_merged.values)
torch_tensor_val = torch.tensor(val_merged.values)

train_dataset = torch_tensor_train.float()
test_dataset = torch_tensor_test.float()
val_dataset = torch_tensor_val.float()
"""
stru_names = data_PDF['stru'].unique().tolist()
#person_values = []
person_values = pd.DataFrame(columns=stru_names)

for indi_stru in stru_names:
    test_indi_stru = test_individuelt[indi_stru]
    size_indi_stru = test_indi_stru["size"]
    stru_indi_stru = test_indi_stru["stru"]
    atom_indi_stru = test_indi_stru["atom"]
    first_peak_indi_stru = test_indi_stru["first_peak"]
    test_indi_stru = test_indi_stru.drop(['stru'], axis=1)
    test_indi_stru = test_indi_stru.drop(['size'], axis=1)
    test_indi_stru = test_indi_stru.drop(['atom'], axis=1)
    test_indi_stru = test_indi_stru.drop(['first_peak'], axis=1)
    test_indi_stru = torch.tensor(test_indi_stru.values)
    test_indi_stru = test_indi_stru.float()
    testloader_indi_stru = DataLoader(
        test_indi_stru,
        batch_size=len(test_indi_stru),
        shuffle=False
    )
    output, dimi, size, stru, data = test_PDF_reconstruction(model, testloader_indi_stru, size_indi_stru, stru_indi_stru)
    output = output.data
    dimi = np.array(dimi.data)
    dimi_df = pd.DataFrame(dimi, columns=['Latent_Space_Variable_1','Latent_Space_Variable_2'])

    pearsonr_values_indi = pd.DataFrame()
    for i in range(len(data)):
        pearsonr = scipy.stats.pearsonr(output[i], data[i])[0]
        person = pd.DataFrame([pearsonr], columns=[str(indi_stru)])
        pearsonr_values_indi = pearsonr_values_indi.append([person])

    person_values = person_values.append(pearsonr_values_indi)
    test_individuelt[indi_stru].reset_index(drop=True, inplace=True)
    pearsonr_values_indi.reset_index(drop=True, inplace=True)
    test_individuelt[indi_stru] = pd.concat([test_individuelt[indi_stru], pearsonr_values_indi], axis=1, join='inner')
    test_individuelt[indi_stru].rename(columns={str(indi_stru): 'pearsonr'}, inplace=True)
    test_individuelt[indi_stru] = pd.concat([test_individuelt[indi_stru], dimi_df], axis=1, join='inner')


means = person_values.mean(skipna=True)
min = person_values.min(skipna=True)
max = person_values.max(skipna=True)
pearsonr_values_pandas = pd.DataFrame({'Mean': means.values, 'Min': min.values, 'Max': max.values}, index=min.index)
pearsonr_values_pandas.to_csv('/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/person_values.csv')

for indi_stru in stru_names:
    test_indi_stru = test_individuelt[indi_stru]
    lowest_3 = test_indi_stru.nsmallest(3, 'pearsonr')
    lowest_3_pdf = lowest_3.iloc[:, :301]
    lowest_tensor = torch.tensor(lowest_3_pdf.values)
    lowest_tensor = lowest_tensor.float()
    lowest_tensor = DataLoader(
        lowest_tensor,
        batch_size=len(lowest_tensor),
        shuffle=False
    )
    output, dimi, size, stru, data = test_PDF_reconstruction(model,lowest_tensor, size_indi_stru, stru_indi_stru)
    output = output.data
    plt.figure()
    plt.plot(data[0])
    plt.plot(output[0])
    plt.plot(data[1]-1)
    plt.plot(output[1]-1)
    plt.plot(data[2]-2)
    plt.plot(output[2]-2)
    a = lowest_3.pearsonr.values[0]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a))
    a2 = lowest_3.pearsonr.values[1]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a2))
    a3 = lowest_3.pearsonr.values[2]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a3))
    plt.title(" Three lowest pearson correlations for test_{}".format(indi_stru))
    plt.xlabel('r [$\AA$]')
    plt.ylabel('G(r)')
    plt.legend(loc= 'upper right')
    plt.savefig(test_path + '/lowest_pearson_correlations correlations_{}.png'.format(indi_stru))

for indi_stru in stru_names:
    test_indi_stru = test_individuelt[indi_stru]
    highest_3 = test_indi_stru.nlargest(3, 'pearsonr')
    highest_3_pdf = highest_3.iloc[:, :301]
    highest_tensor = torch.tensor(highest_3_pdf.values)
    highest_tensor = highest_tensor.float()
    highest_tensor = DataLoader(
        highest_tensor,
        batch_size=len(highest_tensor),
        shuffle=False
    )
    output, dimi, size, stru, data = test_PDF_reconstruction(model,highest_tensor, size_indi_stru, stru_indi_stru)
    output = output.data
    plt.figure()
    plt.plot(data[0])
    plt.plot(output[0])
    plt.plot(data[1]-1)
    plt.plot(output[1]-1)
    plt.plot(data[2]-2)
    plt.plot(output[2]-2)
    a = highest_3.pearsonr.values[0]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a))
    a2 = highest_3.pearsonr.values[1]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a2))
    a3 = highest_3.pearsonr.values[2]
    plt.plot([], [], ' ', label="Pearson correlation")
    plt.plot([], [], ' ', label="{:.4f}".format(a3))
    plt.title(" Three highest pearson correlations for test_{}".format(indi_stru))
    plt.xlabel('r [$\AA$]')
    plt.ylabel('G(r)')
    plt.legend(loc= 'upper right')
    plt.savefig(test_path + '/highest_pearson_correlations correlations_{}.png'.format(indi_stru))


#print(test_individuelt['BCC'])
#print(test_individuelt['BCC'].columns)
#print(test_individuelt['BCC']['stru'])
#print(test_individuelt['BCC']['pearsonr'].round(5))
sammen = pd.concat(test_individuelt)
#first_peak = pd.DataFrame(sammen.iloc[: , :300].idxmax(axis = 1), columns={'first_peak'},  dtype='int32')
#sammen = pd.concat([sammen, first_peak], axis=1, join='inner')

#save_dimi_test_indi(test_individuelt=sammen, stru=sammen['atom'], size=sammen['first_peak'], stru_indi='all_atoms_first_peak', path=test_path)


