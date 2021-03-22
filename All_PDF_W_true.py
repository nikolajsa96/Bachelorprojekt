import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import shutil
# Load Data

stru_true_list = []
PDF_list = []

stru_maker_list = ["FCC", "BCC", "SC", "HCP", "Icosahedron", "Decahedron", "Octahedron"]
for i in stru_maker_list:
    path_PDF_xyz = os.listdir('/home/nikolaj/Desktop/Bachelorprojekt/strus/' + i + '/PDF/')
    path_PDF = '/home/nikolaj/Desktop/Bachelorprojekt/strus/' + i + '/PDF/' # Cluster file

    pbar = tqdm(total=len(path_PDF_xyz))

    for filename in path_PDF_xyz:
        if __name__ == '__main__':
            PDF = pd.read_csv(str(path_PDF) + filename)
            r, Gr = PDF['0'], PDF['1']
            Gr -= np.min(Gr) * 2
            Gr /= np.max(Gr)
            PDF_list.append(Gr)
        PDF_array = np.array(PDF_list)
        pbar.update(1)
    pbar.close()


true = np.genfromtxt('/home/nikolaj/Desktop/Bachelorprojekt/stru_true_list.txt', delimiter=',', dtype=str)
all_PDF = pd.DataFrame(PDF_array)
all_PDF['true']= true

path = (r'/home/nikolaj/Desktop/Bachelorprojekt/strus/' + 'all_PDFs.csv')
all_PDF.to_csv(path, index=False)
