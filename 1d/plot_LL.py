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
deg = 8
L=10
nelts=10*L   # number of elements on interval
npts=10*nelts # for plotting functions
x0=0
x1=L
nc=5 # nc # of p.w. constant 
nn=101
s=10.0 # shift s for landscape 
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
coeftype='random displacement' #'pc'
dmax=0.2
a0=1
a1=100
np.random.seed(5)
#coeftype='constant'
params=''
# create directory and filenames for output
outdir = 'Results/000089'
# filenames
coefplotfile = outdir + '/LL/' + 'coefficient.png'
eigenvalfile = outdir + '/' + 'eigenvalues.txt'
eigens=np.loadtxt(eigenvalfile)
eigenfunh5file  = outdir + '/h5_file/' + 'eigenfun{:05d}.h5'
LLmon = outdir+'/LL/'+'montage{:03d}_{:03d}.png'
Landplotfile=outdir+'/LL/'+'Landscape{:05d}.png'
# write parameters to file
mesh = IntervalMesh(nelts, x0, x1)
x , = SpatialCoordinate(mesh)
# define coefficient A
if coeftype=='pc':
   # pw constants alternately equal to a0 or a1
   aelt = 'DG'
   adeg = 0
   aexpr=Function(FunctionSpace(IntervalMesh(nc, x0, x1),aelt,adeg))
   #aval=a0+np.random.rand(nc2)*(a1-a0)
   aval=a0+np.random.binomial(size=nc, n=1, p= 0.5)*(a1-a0)
   aexpr.vector().set_local(aval)
   V=FunctionSpace(mesh,aelt,adeg)
   A = assemble(interpolate(aexpr, V))
else:
   s=0.25
   ncenter=x1-x0-1
   if coeftype == 'random displacement':
      dn=-dmax+np.random.rand(ncenter+1)*(2*dmax)
   elif coeftype == 'fixed displacement':
      dn=np.zeros(ncenter+1)
   f_sum=0.0
   for i in range(ncenter):
      x_center=i+1+dn[i]
      f_sum=f_sum + conditional(abs(x-x_center)>s,0,6/8*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**3-6/(8*pow(s,2))*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**2*(3*pow(x-x_center,2)+1))
   # near x=x0
   x_center=x0+dn[ncenter]
   f_sum=f_sum + conditional(abs(x-x_center)>s,0,6/8*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**3-6/(8*pow(s,2))*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**2*(3*pow(x-x_center,2)+1))
   # near x=x1
   x_center=x_center+x1-x0
   f_sum=f_sum + conditional(abs(x-x_center)>s,0,6/8*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**3-6/(8*pow(s,2))*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**2*(3*pow(x-x_center,2)+1))
   aexpr=1./(1+f_sum)
   aelt='CG'
   adeg=8
   A = assemble(interpolate(aexpr, FunctionSpace(mesh, aelt, adeg)))
# evaluate coefficient, save to file and plot
plt.clf()
print("> evaluating coefficient")
pts = np.linspace(x0, x1, npts, endpoint=True)
avals = eval_u(A,pts)
plt.plot(pts, avals, alpha=.75, linewidth=2)
plt.xlim([x0, x1])
plt.title('coefficient')
plt.savefig(coefplotfile, dpi=500)
print("> coefficient plotted to {}".format(coefplotfile))



uh=get_shifted_landscape(mesh,x0,x1,s,A,deg,npts,Landplotfile)

for i in range(nn):
     with CheckpointFile(eigenfunh5file.format(i),'r') as afile:
          mesh0=afile.load_mesh()
          eigenfun=afile.load_function(mesh0,"eigen")
     V=FunctionSpace(mesh0,'Lagrange',deg)
     A_int=assemble(interpolate(A, V))
     Aphi=assemble(interpolate(eigenfun*sqrt(A_int), V))
     pts = np.linspace(x0, x1, npts, endpoint=False)
     phi_max=max(abs(eval_u(Aphi,pts)))
     y = eval_u(eigenfun,pts)/((eigens[i]+s)*phi_max)
     # plot (shifted) landscape function 
     plt.clf()
     avals = eval_u(uh,pts)
     plt.plot(pts, avals, alpha=.75, linewidth=0.5, label='u_s')
     plt.plot(pts,y,alpha=.75, linewidth=0.5,label='phi_n')
     plt.legend(loc="lower left")
     plt.xlim([x0, x1])
     plt.title('s={}, eigenfunction {} with eigenvalues {:7.5f}'.format(s,i,eigens[i]))
     plt.savefig(Landplotfile.format(i), dpi=600)
     PETSc.Sys.Print("> LL plotted to {}".format(Landplotfile.format(i)))

ncol_plot=5
nplot_total=ncol_plot*5
dd,dd2=divmod(nn,nplot_total)
for i in range(dd):
    segment=list(range(nplot_total*i,nplot_total*i+nplot_total))
    i0=segment[0]
    iend=segment[-1]
    eigenf_imgs=[]
    for j in segment:
        eigenf_imgs.append(Landplotfile.format(j))
    combine_images(columns=ncol_plot, space=20, images=eigenf_imgs,file=LLmon.format(i0,iend))
    print("> eigenfunction montage between {} and {} is  written to {}".format(i0,iend,LLmon.format(i0,iend) ))
