"""
read h5 file and plot resutls
"""
import random
import csv
from collections import defaultdict
import os,math, json
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from slepc4py import SLEPc
import numpy as np
from solver import *
nreq=51
plotefuns=[int(d) for d in range(nreq)]
# create directory and filenames for output
outdir ='Results/000040'
# filenames
coefplotfile = outdir + '/' + 'coefficient.png'
coefh5file = outdir + '/h5_file/' + 'coefficient.h5'
eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
eigenfunh5file = outdir + '/h5_file/' + 'target_{:05d}_eigen{:05d}.h5'
eigenfunplotfile = outdir + '/' + 'target_{:05d}_eigen{:05d}.png'
eigenfunmontagefile = outdir + '/'+'eigenfunmontage_target_{:05d}_mode_{:03d}.png'

# plot coefficient, save to file
with CheckpointFile(coefh5file,'r') as afile:
     mesh=afile.load_mesh()
     A=afile.load_function(mesh,"coef")
plt.clf()
fig, axes = plt.subplots()
collection = tripcolor(A, axes=axes)
fig.colorbar(collection);
plt.title('coefficient')
plt.savefig(coefplotfile, dpi=500)
PETSc.Sys.Print("> coefficient plotted to {}".format(coefplotfile))

columns = defaultdict(list) # each value in each column is appended to a list
with open(eigen_pratiofile) as f:
    fieldnames =['target','modes','eigenvalue','participation_ratio']
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(np.float128(v)) # append the value into the appropriate list

targets_list=np.array(columns['target']).astype(int)
modes_list=np.array(columns['modes']).astype(int)
eigenvalues_list=np.array(columns['eigenvalue'])
pratio_list=np.array(columns['participation_ratio'])
eigenf_imgs=[]

for j in range(len(targets_list)):
   target=targets_list[j]
   modes=modes_list[j]
   eigen=eigenvalues_list[j]
   pr=pratio_list[j]
   with CheckpointFile(eigenfunh5file.format(target,modes),'r') as afile:
      mesh=afile.load_mesh()
      eigenfun=afile.load_function(mesh,"eigen")
   plt.clf()
   fig, axes = plt.subplots()
   collection = tripcolor(eigenfun, axes=axes)
   collection.set_clim(-1.1,1.1)
   fig.colorbar(collection);
   plt.title(' eigenfunction {}  $\lambda=${:7.5f}  ratio {:1.5f}'.format( modes, eigen,pr))
   print("> eigenfunction {} plotted to ".format(modes) + eigenfunplotfile.format(target,modes))
   plt.savefig(eigenfunplotfile.format(target,modes), dpi=300)
   plt.close()
   eigenf_imgs.append(eigenfunplotfile.format(target,modes))

dd,dd2=divmod(nreq,25)
for i in range(dd):
   segment=list(range(25*i,25*i+25))
   i0=segment[0]
   iend=segment[-1]
   eigenf_imgs=[]
   for j in segment:
      eigenf_imgs.append(eigenfunplotfile.format(target,j))
   combine_images(columns=5, space=20, images=eigenf_imgs,file=eigenfunmontagefile.format(target,i0))
   print("> eigenfunction montage between {} and {} is  written to {}".format(i0,iend,eigenfunmontagefile.format(target,i0)))
