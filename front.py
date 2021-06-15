import sys
import matplotlib as mpl
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats
import mendeleev
import pandas as pd
from mendeleev import element
from scipy.signal import find_peaks

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
    plt.title("Latent Space for {}".format(stru_indi))
    plt.xlabel("Latent Space Variable 1")
    plt.ylabel("Latent Space Variable 2")
    plt.yticks([])
    plt.xticks([])
    plt.xlim(-0.95,0.3)
    plt.ylim(-1.35,0.5)
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
    plt.savefig(path + '/latant_space_plot_{}.png'.format(stru_indi), dpi=200)

path = '/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/test'

test_data = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/test_dataframes.csv")
del test_data[test_data.columns[0]]
del test_data[test_data.columns[0]]
first_peak = []
for i in range(len(test_data)):
    x = test_data.iloc[:, :300]
    x = np.array(x.iloc[i])
    peaks, _ = find_peaks(x, height=0.5)
    first_peak_i = peaks[0]
    first_peak.append(first_peak_i)

test_data = test_data.drop(['first_peak'], axis=1)

test_data['first_peak'] = first_peak
train_data = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/train_dataframes.csv")
del train_data[train_data.columns[0]]
del train_data[train_data.columns[0]]
train_individuelt = dict()
test_individuelt = dict()
for k, v in train_data.groupby('stru'):
    train_individuelt[k] = v
for k, v in test_data.groupby('stru'):
    test_individuelt[k] = v
stru_names = train_data['stru'].unique().tolist()



"""
print(stru_names)
test_indi_stru = test_individuelt['Octahedron'].reset_index()
print(test_indi_stru['pearsonr'].min())
print(test_indi_stru['pearsonr'].max())
print(test_indi_stru['pearsonr'].mean())


test_indi_stru = test_individuelt['FCC'].reset_index()
HCP = train_individuelt['HCP'].reset_index()
SC = train_individuelt['SC'].reset_index()
"""
low_test_data = test_data.nsmallest(3, 'pearsonr')
rcon_start = low_test_data.columns.get_loc("recon_r_0")
rcon_slut = low_test_data.columns.get_loc("recon_r_300")
recon_pdf = low_test_data.iloc[:,(rcon_start):rcon_slut+1]
recon_pdf = recon_pdf
recon_pdf = np.array(recon_pdf)
PDF = low_test_data.iloc[:,0:301]
PDF = PDF
PDF = np.array(PDF)
person = low_test_data['pearsonr']
person = np.array(person)
#stru = low_test_data['stru'][0]
rrange = np.arange(0, 30.1, 0.1)

plt.figure(dpi=200)
plt.subplots(nrows=1, figsize=(12, 9))
plt.xlabel('r [Å]', fontsize= 12)
plt.ylabel('G(r) [a.u]', fontsize= 12)
plt.plot(rrange, PDF[0], 'r', label='Original PDF', linewidth=4)
plt.plot(rrange, recon_pdf[0], 'b',  label='Decoded PDF, pearson: {}'.format(person[0].round(3)),  linewidth=1)
plt.plot(rrange, PDF[1]-1, 'r', label='Original PDF', linewidth=4)
plt.plot(rrange, recon_pdf[1]-1, 'b',  label='Decoded PDF, pearson: {}'.format(person[1].round(3)),  linewidth=1)
plt.plot(rrange, PDF[2]-2, 'r', label='Original PDF', linewidth=4)
plt.plot(rrange, recon_pdf[2]-2, 'b',  label='Decoded PDF, pearson: {}'.format(person[2].round(3)),  linewidth=1)

plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(path + '/3_shits.png', dpi=200)





"""
recon_HCP = HCP.iloc[:,(rcon_start):rcon_slut+1]
recon_HCP = recon_HCP.iloc[0]
recon_HCP = np.array(recon_HCP)
PDF_HPC = HCP.iloc[:,1:302]
PDF_HPC = PDF_HPC.iloc[0]
PDF_HPC = np.array(PDF_HPC)
person_HCP = HCP['pearsonr'][0]
stru_HCP = HCP['stru'][0]


recon_pdf_SC = SC.iloc[:,(rcon_start):rcon_slut+1]
recon_pdf_SC = recon_pdf_SC.iloc[4]
recon_pdf_SC = np.array(recon_pdf_SC)
PDF_SC = SC.iloc[:,1:302]
PDF_SC = PDF_SC.iloc[4]
PDF_SC = np.array(PDF_SC)
person_SC = SC['pearsonr'][4]
stru_SC = SC['stru'][4]


plt.figure(dpi=200)
plt.subplots(nrows=1, figsize=(12, 9))
plt.xlabel('r [Å]', fontsize= 12)
plt.ylabel('G(r) [a.u]', fontsize= 12)
plt.plot(rrange, PDF_HPC, 'r', label='Original PDF, {}'.format(stru_HCP), linewidth=4)
plt.plot(rrange, recon_HCP, 'b',  label='Decoded PDF, pearson: {}'.format(person_HCP.round(3)),  linewidth=1)
plt.plot(rrange, recon_HCP-PDF_HPC-1, 'b',  label='Decoded PDF, pearson: {}'.format(person_HCP.round(3)),  linewidth=1)

plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(path + '/haha_funny.png', dpi=200, transparent=True)
"""

from collections import Counter
strunow = np.genfromtxt('/home/nikolaj/Desktop/Bachelorprojekt/strus/stru_true_list.txt',dtype='str')

#print(Counter(strunow))

test_indi_stru = test_individuelt['HCP'].reset_index(drop=True)

test_indi_stru = test_indi_stru[(test_indi_stru == 'Ni').any(axis=1)]
test_indi_stru = test_indi_stru.reset_index(drop=True)

Left_RU = test_indi_stru.nsmallest(7, 'Latent_Space_Variable_1')
Left_RU = Left_RU.reset_index(drop=True)
left_Ru_PDF = Left_RU.iloc[:,0:301]

#print(test_indi_stru['Latent_Space_Variable_1'])
#print(test_indi_stru['Latent_Space_Variable_2'])


"""
plt.figure(dpi=200)
plt.subplots(nrows=1, figsize=(12, 9))
plt.xlabel('r [Å]', fontsize= 12)
plt.ylabel('G(r) [a.u]', fontsize= 12)
for i in range(len(left_Ru_PDF)):
    plt.plot(rrange, left_Ru_PDF.iloc[i]-i, label='{}. left most'.format(i+1), linewidth=4)
plt.legend(fontsize=15, loc='upper right')
plt.tight_layout()
plt.show()
"""

right_RU = test_indi_stru.nlargest(7, 'Latent_Space_Variable_1')
#right_RU = right_RU.iloc[::2, :]
right_RU = right_RU.reset_index(drop=True)
right_Ru_PDF = right_RU.iloc[:,0:301]
plt.figure(dpi=200)
plt.subplots(nrows=1, figsize=(12, 9))
plt.xlabel('r [Å]', fontsize= 12)
plt.ylabel('G(r) [a.u]', fontsize= 12)
for i in np.arange(7):
    plt.plot(rrange, right_Ru_PDF.iloc[i]-i, label='{}. right most'.format(i+1), linewidth=4)
plt.legend(fontsize=15, loc='upper right')

plt.tight_layout()

plt.savefig(path + '/Ni_HCP_1_7png', dpi=200)

"""
plt.figure(dpi=200)
plt.subplots(nrows=1, figsize=(12, 9))
plt.xlabel('r [Å]', fontsize= 12)
plt.ylabel('G(r) [a.u]', fontsize= 12)
for i in np.arange(6,11):
    plt.plot(rrange, right_Ru_PDF.iloc[i]-i, label='{}. right most'.format(i+1), linewidth=4)
plt.legend(fontsize=15, loc='upper right')
plt.tight_layout()
#plt.savefig(path + '/W_HCP_7_11.png', dpi=200)
"""

#right_RU = right_RU.drop([15, 14, 11, 12])
#right_RU = right_RU.reset_index(drop=True)

#save_dimi_test_indi(test_individuelt=Left_RU, stru=Left_RU['atom'], size=Left_RU['atom_radi'], stru_indi='Left_RU', path=path)
#save_dimi_test_indi(test_individuelt=test_data, stru=test_data['atom'], size=test_data['pearsonr'], stru_indi='test_all_atoms_pearson', path=path)
