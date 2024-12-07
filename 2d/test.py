import os,math
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from slepc4py import SLEPc
import numpy as np
from solver import *
deg = 5
L=100 # length of square
nx=100
ny=100
mesh = SquareMesh(nx,ny,L)
x , y= SpatialCoordinate(mesh)
s=0.25
nn=L-1
# unifrom distribution on unit disk
U1 = np.random.uniform(size = nn**2)
U2 = np.random.uniform(size = nn**2)
r = 0.2
dn1 = r * np.sqrt(U2) * np.cos(2 * np.pi * U1)
dn2 = r * np.sqrt(U2) * np.sin(2 * np.pi * U1)
Asum=Function(FunctionSpace(mesh, 'CG', 7))
for i in range(0,nn):
   for j in range(0,nn):
      x_center=[i+1+dn1[i*nn+j],j+1+dn2[i*nn+j]]
      fi= conditional(((x-x_center[0])**2+(y-x_center[1])**2)**0.5>s,0,20*(1-((x-x_center[0])**2+(y-x_center[1])**2)/pow(s,2))**3*(3*((x-x_center[0])**2+(y-x_center[1])**2)+1))
     # Fi=assemble(interpolate(1.0/(1+fi), FunctionSpace(mesh, 'CG', 7)))
     # A+=Fi
      Fi=assemble(interpolate(fi, FunctionSpace(mesh, 'CG', 7)))
      Asum+=Fi
A=assemble(interpolate(Constant(1.0)/(Constant(1.0)+Asum), FunctionSpace(mesh, 'CG', 7)))
#print('defining coefficients')
#A=assemble(interpolate(1.0/(1.0+Fs), FunctionSpace(mesh, 'CG', 3)))
# evaluate coefficient, save to file and plot
plt.clf()
fig, axes = plt.subplots()
collection = tripcolor(A, axes=axes)
fig.colorbar(collection);
plt.title('coefficient')
plt.savefig("coef.png", dpi=500)
print("> coefficient plotted to coef.png")

