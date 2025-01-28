"""
plot checkerboard 2d results  in montage 
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
outdir ='Results/000034'

nn=5 # plot nn x nn figures montage 
eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
eigenfunplotfile_old = outdir + '/' + 'target_0.00_eigen{:05d}.png'
eigenfunplotfile = outdir + '/' + 'eigen{:05d}.png'
eigenfunmontagefile = outdir + '/'+'newmontage.png'
columns = defaultdict(list) # each value in each column is appended to a list
with open(eigen_pratiofile) as f:
    fieldnames =['target','eigenvalue','participation_ratio']
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value 
            columns[k].append(np.float128(v)) # append the value into the appropriate list

eigenvalues_list=np.array(columns['eigenvalue'])
eigenf_imgs=[]
Lx=100
Ly=100
nelts=2001
x=np.linspace(0,Lx,nelts)
y=np.linspace(0,Ly,nelts)
X,Y =np.meshgrid(x,y)
a1=1
a2=10
ne=len(eigenvalues_list)
idx_list=[i for i in range(ne)]
def pc_function(X,Y,nelts,a1,a2,i0,j0,Lx,Ly):
    Z=np.zeros_like(X)
    for i in range(nelts):
       for j in range(nelts):
           if  ((0<=X[i,j]<=Lx/2) and (0<=Y[i,j]<=Ly/2)) or ((Lx/2<=X[i,j]<=Lx) and (Ly/2<=Y[i,j]<=Ly)):
                Z[i,j]=np.sin((i0+1)*np.pi*X[i,j]/Lx)*np.sin((j0+1)*np.pi*Y[i,j]/Ly)
           else:
                Z[i,j]=a1/a2*np.sin((i0+1)*np.pi*X[i,j]/Lx)*np.sin((j0+1)*np.pi*Y[i,j]/Ly)
    return Z
def closest(list, Number):
    aux = []
    for value in list:
        aux.append(abs(Number-value))
    return aux.index(min(aux))
for i in range(nn):
   for j in range(nn):
      if ((i+1)%2==1 and (j+1)%2==1) or ((i+1)%2==0 and (j+1)%2==0):
         eigen=(np.pi)**2*(((i+1)/Lx)**2+((j+1)/Ly)**2)
         eigen_list=eigenvalues_list[idx_list]
         idex=closest(eigen_list,eigen)
         del idx_list[idex]
for i in range(nn):
   for j in range(nn):
      if (i+1)%2==1 and (j+1)%2==1:
         eigen=(np.pi)**2*(((i+1)/Lx)**2+((j+1)/Ly)**2)
         Z = np.sin((i+1)*np.pi*X/Lx)*np.sin((j+1)*np.pi*Y/Ly)
         plt.clf()  
         fig, axes = plt.subplots()
         im = plt.pcolormesh(X,Y,Z,cmap='bwr')
         axes.set_aspect('equal') 
         im.set_clim(-1.1,1.1)
         plt.xlabel("x")
         plt.ylabel("y")
         fig.colorbar(im)
         plt.title(' eigenfunction {},{}  $\lambda=${:7.5f}'.format(j+1,i+1,eigen))
         plt.savefig(eigenfunplotfile.format(i*nn+j),dpi=300)
         plt.close()
         eigenf_imgs.append(eigenfunplotfile.format(j*nn+i))
      elif (i+1)%2==0 and (j+1)%2==0: 
         eigen=(np.pi)**2*(((i+1)/Lx)**2+((j+1)/Ly)**2)
         Z=pc_function(X,Y,nelts,a1,a2,i,j,Lx,Ly)
         plt.clf()  
         fig, axes = plt.subplots()
         im = plt.pcolormesh(X,Y,Z,cmap='bwr')
         axes.set_aspect('equal') 
         im.set_clim(-1.1,1.1)
         plt.xlabel("x")
         plt.ylabel("y")
         fig.colorbar(im)
         plt.title(' eigenfunction {},{}  $\lambda=${:7.5f}'.format(j+1,i+1,eigen))
         plt.savefig(eigenfunplotfile.format(i*nn+j),dpi=300)
         plt.close()
         eigenf_imgs.append(eigenfunplotfile.format(j*nn+i))
      else:
         target=(np.pi)**2*(((i+1)/Lx)**2+((j+1)/Ly)**2)
         eigen_list=eigenvalues_list[idx_list]
         idex=closest(eigen_list,target)
         eigen=eigen_list[idex]
         n_eigen=idx_list[idex]
         eigenf_imgs.append(eigenfunplotfile_old.format(n_eigen))
         del idx_list[idex]
combine_images(columns=nn, space=20, images=eigenf_imgs,file=eigenfunmontagefile)
print("> eigenfunction montage is  written to {}".format(eigenfunmontagefile))
