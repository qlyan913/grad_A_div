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
outdir ='Results/000033'

nn=5 # plot nn x nn figures montage 
eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
eigenfunplotfile_old = outdir + '/' + 'target_0.00_eigen{:05d}.png'
eigenfunplotfile = outdir + '/' + 'eigen{:05d}.png'
eigenfunmontagefile = outdir + '/'+'newmontage.png'
columns = defaultdict(list) # each value in each column is appended to a lis
eigenvalues_list=np.array(columns['eigenvalue'])
eigenf_imgs=[]
L=100
nelts=2001
x=np.linspace(0,L,nelts)
y=np.linspace(0,L,nelts)
X,Y =np.meshgrid(x,y)
a1=1
a2=10
ne=len(eigenvalues_list)
idx_list=[i for i in range(ne)]
def pc_function(X,Y,nelts,a1,a2,i0,j0):
    Z=np.zeros_like(X)
    for i in range(nelts):
       for j in range(nelts):
           if  0<=Y[i,j]<=L/2:
                Z[i,j]=np.sin((i0+1)*np.pi*X[i,j]/L)*np.sin((j0+1)*np.pi*Y[i,j]/L)
           else:
                Z[i,j]=a1/a2*np.sin((i0+1)*np.pi*X[i,j]/L)*np.sin((j0+1)*np.pi*Y[i,j]/L)
    return Z

for i in range(nn):
   for j in range(nn):
      if (j+1)%2==1:
         eigen=(np.pi/L)**2*((i+1)**2+(j+1)**2)
         Z = np.sin((i+1)*np.pi*X/L)*np.sin((j+1)*np.pi*Y/L)
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
         eigen=(np.pi/L)**2*((i+1)**2+(j+1)**2)
         Z=pc_function(X,Y,nelts,a1,a2,i,j)
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

combine_images(columns=nn, space=20, images=eigenf_imgs,file=eigenfunmontagefile)
print("> eigenfunction montage is  written to {}".format(eigenfunmontagefile))
