import os, random, sys
import matplotlib.pyplot as plt
import numpy as np
from diffpy.Structure import loadStructure, Structure, Lattice
from diffpy.srreal.pdfcalculator import DebyePDFCalculator, PDFCalculator
from tqdm import tqdm
import pandas as pd
import shutil

sys.path.append(os.getcwd())
random.seed(14)  # 'Random' numbers


class PDF_generator:
    def __init__(self, qmin=0.5, qmax=30, qdamp=0.03, rmin=0, rmax=30, rstep=0.1, biso=0.3, delta2=0):
        """
        Class for simulating total scattering data for both crystal and cluster files.

        Parameters
        ----------
        qmin : Smallest q-vector included (float)
        qmax : Instrumental resolution (float)
        qdamp : Instrumental dampening (float)
        rmin : Start of r-grid (float)
        rmax : End of r-grid (float)
        rstep : Spacing between r-grid (float)
        biso : Vibration (float (float)
        delta2 : Correlated vibration (float)
        """

        self.qmin = qmin
        self.qmax = qmax  # Instrument resolution
        self.qdamp = qdamp  # Instrumental dampening
        self.rmin = rmin  # Smallest r value
        self.rmax = rmax  # Can not be less then 10 AA
        self.rstep = rstep  # Nyquist for qmax = 30
        self.biso = biso  # Atomic vibration
        self.delta2 = delta2  # Corelated vibration


    def genPDF(self, clusterFile, fmt='cif'):
        """
        Simulates PDF for input structure.

        Parameters
        ----------
        clusterFile : path for input file (str)
        fmt : cif or xyz, depending on structure type (str)

        Returns
        -------
        r : r-grid (NumPy array)
        Gr : Intensities (NumPy array)
        """
        stru = loadStructure(clusterFile)

        stru.B11 = self.biso
        stru.B22 = self.biso
        stru.B33 = self.biso
        stru.B12 = 0
        stru.B13 = 0
        stru.B23 = 0

        if fmt.lower() == 'cif':
            PDFcalc = PDFCalculator(rmin=self.rmin, rmax=self.rmax, rstep=self.rstep,
                                    qmin=self.qmin, qmax=self.qmax, qdamp=self.qdamp, delta2=self.delta2)
        elif fmt.lower() == 'xyz':
            PDFcalc = DebyePDFCalculator(rmin=self.rmin, rmax=self.rmax, rstep=self.rstep,
                                         qmin=self.qmin, qmax=self.qmax, qdamp=self.qdamp, delta2=self.delta2)
        r0, g0 = PDFcalc(stru)

        self.r = np.array(r0)
        self.Gr = np.array(g0)

        return self.r, self.Gr

    def __str__(self):
        return 'PDF parameters:\n\tqmin: {}\n\tqmax: {}\n\tqdamp: {}\n\trmin: {}' \
                              '\n\trmax {}\n\trstep: {}\n\tbiso: {} \n\tDelta2: {}\n'\
                                        .format(self.qmin, self.qmax, self.qdamp, self.rmin,
                                                self.rmax, self.rstep, self.biso, self.delta2)
stru_true_list = []
size_list = []
stru_maker_list = ["FCC", "BCC", "SC", "HCP", "Icosahedron", "Decahedron", "Octahedron"]
for i in stru_maker_list:
    dir = '/home/nikolaj/Desktop/Bachelorprojekt/strus/' + i + '/PDF/'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    path_tmp_xyz = os.listdir('/home/nikolaj/Desktop/Bachelorprojekt/strus/' + i + '/xyz/')
    path_xyz = '/home/nikolaj/Desktop/Bachelorprojekt/strus/' + i + '/xyz/' # Cluster file

    pbar = tqdm(total=len(path_tmp_xyz))

    for filename in path_tmp_xyz:
        with open(path_xyz + filename) as f:
            first_line = int(f.readline())
        if __name__ == '__main__':
            PDF_obj = PDF_generator()  # Init class object
            # Using cluster files
            xyz_path =str(path_xyz) + filename
            xgrid, xyz_pdf = PDF_obj.genPDF(xyz_path, fmt='xyz')
            PDF_xyz = pd.DataFrame([xgrid, xyz_pdf])
            PDF_xyz = PDF_xyz.T
            new_path = (r'/home/nikolaj/Desktop/Bachelorprojekt/strus/' + i + '/PDF/'+'PDF_' + filename)
            new_path = new_path.replace('.xyz', '.cvs')
            PDF_xyz.to_csv(new_path, index=False)
            base = os.path.basename(filename)
            txt = os.path.splitext(base)[0]
            x = txt.split("_")
            stru_true_list.append(x[0])
            size_list.append((first_line))
        pbar.update(1)
    pbar.close()
np.savetxt("stru_true_list.txt", stru_true_list, fmt='%.3s', delimiter=',')
np.savetxt("size_list.txt", size_list, delimiter=',')
"""

os.makedirs("/home/nikolaj/Desktop/Bachelorprojekt/strus/FCC/PDF/",exist_ok=True)
path_tmp_xyz=os.listdir('/home/nikolaj/Desktop/Bachelorprojekt/strus/FCC/xyz/')
pbar = tqdm(total=len(path_tmp_xyz))

for filename in path_tmp_xyz:
    if __name__ == '__main__':
        xyz_path = '/home/nikolaj/Desktop/Bachelorprojekt/strus/FCC/xyz/'+ filename  # Cluster file

        PDF_obj = PDF_generator()  # Init class object
        # Using cluster files
        xgrid, xyz_pdf = PDF_obj.genPDF(xyz_path, fmt='xyz')
        xgrid_list = [xgrid]
        xyz_pdf_list = [xyz_pdf]
        PDF_xyz = pd.DataFrame(list(zip(xgrid_list, xyz_pdf_list)))
        new_path = (r'/home/nikolaj/Desktop/Bachelorprojekt/strus/FCC/PDF/'+'PDF'+filename)
        new_path = new_path.replace('.xyz', '.cvs')
        PDF_xyz.to_csv(new_path, index=False)
    pbar.update(1)
pbar.close()
"""