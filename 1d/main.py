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
nelts=3000
npts=3000
x0=0
x1=10
nreq=1000
target=0
#plotefuns=0,10,20,30,40,50,100,200,300,400,500,600,700,800,900,999
plotefuns=[int(d) for d in range(20)]
bctype='dirichlet' # dirichlet or neumann
coeftype='1d random displacement'
params=''
# create directory and filenames for output
outdir = makedir()
# filenames
coefplotfile = outdir + '/' + 'coefficient.png'
eigenvalfile = outdir + '/' + 'eigenvalues.txt'
eigenfunplotfile = outdir + '/' + 'eigenfun{:05d}.png'
eigenfunmontagefile = outdir + '/'+'eigenfunmontage.png'
paramfile = outdir+ '/'+'Parameter.json'

# write parameters to file
# store parameters in dictionary
runparameters = {
    'bctype': bctype,
    'deg': deg,
    'nelts': nelts,
    'npts': npts,
    'params': params,
    'coeftype': coeftype,
    'x0': x0,
    'x1': x1,
    }
paramf = open(paramfile, 'w')
json.dump(runparameters, paramf, indent=4)
paramf.write('\n')
paramf.close()
print("> run parameters written to {}".format(paramfile))

mesh = IntervalMesh(nelts, x0, x1)
x , = SpatialCoordinate(mesh)
# define coefficient A
dmax=0.2
s=0.25
nn=x1-x0-1
dn=-dmax+np.random.rand(nn)*(2*dmax)
f_sum=0.0
for i in range(nn):
   x_center=i+1+dn[i]
   f_sum=f_sum + conditional(abs(x-x_center)>s,0,6/8*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**3-6/(8*pow(s,2))*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**2*(3*pow(x-x_center,2)+1))
aexpr=1./(1+f_sum)
aelt='CG'
adeg=3
#aval=1
#aexpr = Constant(aval)
#aelt = 'DG'
#adeg = 0
A = assemble(interpolate(aexpr, FunctionSpace(mesh, aelt, adeg)))
# evaluate coefficient, save to file and plot
plt.clf()
print("> evaluating coefficient")
pts = np.linspace(x0, x1, npts, endpoint=True)
avals = eval_u(A,pts)
plt.plot(pts, avals, alpha=.75, linewidth=2)
plt.xlim([x0, x1])
plt.title('coefficient'.format(nelts, deg))
plt.savefig(coefplotfile, dpi=500)
print("> coefficient plotted to {}".format(coefplotfile))

# solve eigen problem and save results
EPS, nconv, Bsc, V=eigen_solver(mesh,A,deg,nreq,target,bctype)
get_eigenpairs(EPS,nconv,Bsc,V,x0,x1,nelts,npts,plotefuns,eigenvalfile,eigenfunplotfile,eigenfunmontagefile)
