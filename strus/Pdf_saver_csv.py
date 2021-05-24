import os
import sys

import pandas as pd
import numpy as np
import shutil
import tqdm
import glob

stru_true_list = []
pdf_list = []
atom_list = []
siz_list = np.loadtxt(r'/home/nikolaj/Desktop/Bachelorprojekt/strus/size_list.txt')

stru_maker_list = ["FCC",
                   "BCC",
                   "SC",
                   "HCP",
                   "Icosahedron",
                   "Decahedron",
                   "Octahedron"
                    ]

dir = os.listdir('/home/nikolaj/Desktop/Bachelorprojekt/xyz_db_raw_atoms_200_interpolate_001/all_PDF')
prat = '/home/nikolaj/Desktop/Bachelorprojekt/xyz_db_raw_atoms_200_interpolate_001/all_PDF/'
for filename in dir:
    base = os.path.basename(filename)
    txt = os.path.splitext(base)[0]
    x = txt.split("_")
    index = x.index("atom") + 1
    atom_list.append(str(x[index]))
    stru_true_list.append(x[1])
    PDF = np.loadtxt(prat + base, delimiter=',')
    r, Gr = PDF[:, 0], PDF[:, 1]
    Gr = (Gr - np.min(Gr)) / (np.max(Gr) - np.min(Gr))
    pdf_list.append(Gr)

all_PDF = pd.DataFrame(pdf_list)

column_maxes = all_PDF.max()
df_max = column_maxes.max()
column_min = all_PDF.min()
df_min = column_min.min()
all_PDF_nord = (all_PDF - df_min) / (df_max - df_min)

all_PDF_nord['stru'] = stru_true_list
all_PDF_nord['size'] = siz_list
all_PDF_nord['atom'] = atom_list
os.makedirs('/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/', exist_ok=True)
path = (r'/home/nikolaj/Desktop/Bachelorprojekt/pdf_notsammen/' + 'all.csv')
all_PDF_nord.to_csv(path, index=False)

print(type(atom_list[0]))
print(all_PDF_nord.dtypes)