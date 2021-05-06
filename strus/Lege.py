import os
import pandas as pd
import numpy as np
import shutil
import tqdm
import glob

stru_true_list = []
pdf_list = []
stru_maker_list = [#"FCC",
                   "BCC",
                   #"SC",
                   #"HCP",
                   "Icosahedron"
                   #"Decahedron",
                   #"Octahedron"
                    ]
for i in stru_maker_list:
    dir = os.listdir('/home/nikolaj/Desktop/Bachelorprojekt/Sigsig_data/' + i + '/')
    prat = '/home/nikolaj/Desktop/Bachelorprojekt/Sigsig_data/' + i + '/'
    for filename in dir:
        base = os.path.basename(filename)
        txt = os.path.splitext(base)[0]
        x = txt.split("_")
        stru_true_list.append(x[1])
        PDF = np.loadtxt(prat + base, delimiter=',')
        r, Gr = PDF[:, 0], PDF[:, 1]
        Gr = (Gr - np.min(Gr)) / (np.max(Gr) - np.min(Gr))
        pdf_list.append(Gr)


all_PDF = pd.DataFrame(pdf_list)
all_PDF['true'] = stru_true_list

path = (r'/home/nikolaj/Desktop/Bachelorprojekt/Sigsig_data/' + 'BCC_iso.csv')
all_PDF.to_csv(path, index=False)