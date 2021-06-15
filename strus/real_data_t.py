import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn.functional as F
import torch
import torch.nn as nn
import pandas as pd
import scipy.stats
from scipy.signal import find_peaks


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


def save_dimi_test_indi(test_individuelt, stru, size, stru_indi, dimi, path, dimi2, dimi3):
    norm = plt.Normalize(1, 4)
    c = np.array(size) / 100
    names = stru
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, cmap.N)
    fig, ax = plt.subplots(figsize=(5, 8))
    sc = ax.scatter(test_individuelt['Latent_Space_Variable_1'], test_individuelt['Latent_Space_Variable_2'], c=c, cmap=cmap, s=80)
    plt.title("Latent Space for {}".format(stru_indi))
    plt.xlabel("Latent Space Variable 1")
    plt.ylabel("Latent Space Variable 2")
    plt.yticks([])
    plt.xticks([])
    #plt.xlim(-0.95,0.3)
    #plt.ylim(-1.35,0.5)
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
        ax.annotate(stru[iter][:3], xy=(i, j), color='black',
                    fontsize="x-small", weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')

    annot.set_visible(False)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.scatter(dimi[0], dimi[1], c='r', s=25,label='JQ_S3_Pt_FCC.gr')
    plt.scatter(dimi2[0], dimi2[1], c='b', s=25, label='pd.gr')
    plt.scatter(dimi3[0], dimi3[1], c='g', s=25,label='Au144SC6.gr')
    plt.legend()
    plt.savefig(path + '/latant_space_plot_{}.png'.format(stru_indi), dpi=200)

model = Net()
model = torch.load('/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/trainfull_model.pt')
model.eval()

test_data = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/test_dataframes.csv")
train_data = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/train_dataframes.csv")
del test_data[test_data.columns[0]]
del test_data[test_data.columns[0]]

#print(test_data['atom'].unique())

data_1 = np.loadtxt('/home/nikolaj/Desktop/Bachelorprojekt/JQ_S3_Pt_FCC.gr', skiprows=27)
data_2 = np.loadtxt('/home/nikolaj/Desktop/Bachelorprojekt/Pd.gr', skiprows=27)
data_3 = np.loadtxt('/home/nikolaj/Desktop/Bachelorprojekt/Au144SC6.gr', skiprows=22)
r,Gr = data_1[:,0],data_1[:,1]
r,Gr2 = data_2[:,0],data_2[:,1]
r,Gr3 = data_3[:,0],data_3[:,1]

Gr = Gr[::10]
Gr = Gr[:301]
Gr2 = Gr2[::10]
Gr2 = Gr2[:301]
Gr3 = Gr3[::10]
Gr3 = Gr3[:301]
Gr3 = np.pad(Gr3, (0, 1), 'constant')
"""
print(len(Gr))
plt.plot(Gr)
plt.show()
sys.exit()
"""
Gr = (Gr - np.min(Gr)) / (np.max(Gr) - np.min(Gr))
Gr2 = (Gr2 - np.min(Gr2)) / (np.max(Gr2) - np.min(Gr2))
Gr3 = (Gr3 - np.min(Gr3)) / (np.max(Gr3) - np.min(Gr3))
rrange = np.arange(0, 30.1, 0.1)
Gr = torch.tensor(Gr)
Gr = Gr.float()
Gr2 = torch.tensor(Gr2)
Gr2 = Gr2.float()
Gr3 = torch.tensor(Gr3)
Gr3 = Gr3.float()
output = model(Gr)
dimi = output[1]
dimi = dimi.cpu().detach().numpy()
output = output[0]

output2 = model(Gr2)
dimi2 = output2[1]
dimi2 = dimi2.cpu().detach().numpy()
output2 = output2[0]

output3 = model(Gr3)
dimi3 = output3[1]
dimi3 = dimi3.cpu().detach().numpy()
output3 = output3[0]


path = '/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/test'
#save_dimi_test_indi(test_individuelt=test_data, stru=test_data['atom'], size=test_data['atom_radi'], stru_indi='real_data', dimi=dimi, path=path, dimi2=dimi2, dimi3=dimi3)

output = output.cpu().detach().numpy()
output2 = output2.cpu().detach().numpy()
output3 = output3.cpu().detach().numpy()


"""
plt.figure()
pearsonr = scipy.stats.pearsonr(output, Gr)[0]
plt.plot(rrange,Gr, 'r', label='JQ_S3_Pt_FCC.gr',linewidth=3)
plt.plot(rrange,output, 'y', label='Decoded, Pearson:{}'.format(pearsonr.round(3)), linewidth=1)
pearsonr2 = scipy.stats.pearsonr(output2, Gr2)[0]
plt.plot(rrange,Gr2-1, 'b', label='Pd.gr',linewidth=3)
plt.plot(rrange,output2-1, 'y', label='Decoded, Pearson:{}'.format(pearsonr2.round(3)), linewidth=1)
pearsonr3 = scipy.stats.pearsonr(output3, Gr3)[0]
plt.plot(rrange,Gr3-2, 'g', label='Au144SC6.gr',linewidth=3)
plt.plot(rrange,output3-2, 'y', label='Decoded, Pearson:{}'.format(pearsonr2.round(3)), linewidth=1)
plt.legend()
plt.title('Reconstruction for real data')
plt.savefig(path + '/real_data.png', dpi=200)
"""




x = np.array(Gr3)
peaks, _ = find_peaks(x, height=0.5)
plt.figure()
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.plot(np.zeros_like(x), "--", color="gray")
plt.show()
print(peaks)