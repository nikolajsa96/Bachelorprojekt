import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats
import mendeleev
import pandas as pd
from mendeleev import element

vej = 'pdf_notsammen'
data_PDF = pd.read_csv("/home/nikolaj/Desktop/Bachelorprojekt/" + vej + "/all.csv")
gay_list=[]
atoms = data_PDF['atom']

for i in range(len(atoms)):
    atom = element(str(atoms[i]))
    atom_radi = atom.atomic_radius
    gay_list.append(atom_radi)
print(gay_list)