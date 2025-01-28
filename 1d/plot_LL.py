"""
Solve the eigenvalue problem with variable coefficient:
   -(Au')'=lambda u on [x0,x1]
Here, we consider the 1d random displacement model:
   A(x) = 1/(1+ sum_{integer n: x0<= n <= x1}f(x-n-dn(w))
   f = 1/8[max{(1-x^2/s^2)^3,0}(3x^2+1)]', supp(f) in [-s,s]
   dn uniform distribution on [-dmax,dmax]
   We choose s=1/4 and dmax=1/5 such that s+dmax<1/2
"""
import os,math, json
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from slepc4py import SLEPc
import numpy as np
from solver import *
deg = 7
L=200
nelts=5*L   # number of elements on interval
npts=2*nelts # for plotting functions
x0=0
x1=L
nc=L # nc # of p.w. constant 
nn=101
ncol_plot=4
s=1.0 # shift s for landscape 
target=0
plotefuns=0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600,700,800,900,999
plotefuns_2=[int(d) for d in range(20)]
flag=1 # print all first nreq  eigenfuns, ignore plotefuns and plotefuns_2
flag2 =2
"""
     flag2 ---- 1: -div A grad phi = lambda phi
          ---- 2: -div A grad phi = lambda A phi
          ---- 3: -div grad phi = lambda A phi
"""
bctype='dirichlet' # dirichlet or neumann
coeftype='pc'
a0=1
a1=10
np.random.seed(5)
#coeftype='constant'
params=''
# create directory and filenames for output
outdir = 'Results/000081'
# filenames
coefplotfile = outdir + '/' + 'coefficient.png'
eigenvalfile = outdir + '/' + 'eigenvalues.txt'
eigens=np.loadtxt(eigenvalfile)
eigenfunh5file  = outdir + '/h5_file/' + 'eigenfun{:05d}.h5'

Landplotfile=outdir+'/'+'Landscape.png'
# write parameters to file
mesh = IntervalMesh(nelts, x0, x1)
x , = SpatialCoordinate(mesh)
# define coefficient A
# pw constants alternately equal to a0 or a1
aelt = 'DG'
adeg = 0
aexpr=Function(FunctionSpace(IntervalMesh(nc, x0, x1),aelt,adeg))
#aval=a0+np.random.rand(nc2)*(a1-a0)
aval=a0+np.random.binomial(size=nc, n=1, p= 0.5)*(a1-a0)
aexpr.vector().set_local(aval)
V=FunctionSpace(mesh,aelt,adeg)
A = assemble(interpolate(aexpr, V))
uh=get_shifted_landscape(mesh,x0,x1,s,A,deg,npts,Landplotfile)
with CheckpointFile(eigenfunh5file.format(nn),'r') as afile:
     mesh0=afile.load_mesh()
     eigenfun=afile.load_function(mesh0,"eigen")
V=FunctionSpace(mesh0,'Lagrange',deg)
A=assemble(interpolate(A, V))
Aphi=assemble(interpolate(eigenfun*sqrt(A), V))
pts = np.linspace(x0, x1, npts, endpoint=False)
phi_max=max(eval_u(Aphi,pts))
y = eval_u(eigenfun,pts)/((eigens[nn]+s)*phi_max)
# plot (shifted) landscape function 
plt.clf()
avals = eval_u(uh,pts)
plt.plot(pts, avals, alpha=.75, linewidth=2, label='u')
plt.plot(pts,y,alpha=.75, linewidth=2,label='phi_n')
plt.xlim([x0, x1])
plt.title('shifted landscape with eigenfunction {}'.format(nn))
plt.savefig(Landplotfile, dpi=500)
PETSc.Sys.Print("> LL plotted to {}".format(Landplotfile))


