"""
Solve the eigenvalue problem with variable coefficient:
   -(Au')'=lambda u
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
nelts=500
npts=2000
x0=0
x1=math.pi
nreq=1000
target=0
plotefuns=0,10,20,50,100,200,300,400,500,600,700,800,900,999
bctype='neumann' # dirichlet or neumann
coeftype='constant'
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
aval=1
aexpr = Constant(aval)
aelt = 'DG'
adeg = 0
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
