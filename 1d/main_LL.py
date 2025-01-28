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
nreq=501
ncol_plot=4
s=0.0 # shift s for landscape 
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
outdir = makedir() # 'Results/000066'
# filenames
coefplotfile = outdir + '/' + 'coefficient.png'
eigenvalfile = outdir + '/' + 'eigenvalues.txt'
mpfile = outdir + '/' + 'pratio_mode.png'
epfile = outdir + '/' + 'pratio_eigen.png'
mpfile_log = outdir + '/' + 'pratio_mode_log.png'
epfile_log = outdir + '/' + 'pratio_eigen_log.png'
pratiofile=outdir + '/' + 'pratio.txt'
eigenfunplotfile = outdir + '/' + 'eigenfun{:05d}.png'
eigenfunh5file  = outdir + '/h5_file/' + 'eigenfun{:05d}.h5'
eigenfunmontagefile = outdir + '/'+'eigenfunmontage.png'
eigenfunmontagefile_2 = outdir + '/'+'eigenfunmontage_v2.png'
eigenfunmon_all = outdir+'/'+'eigenfunmon{:03d}_{:03d}.png'
paramfile = outdir+ '/'+'Parameter.json'
Vplotfile= outdir+'/'+'potential.png'
Landplotfile=outdir+'/'+'Landscape.png'
# write parameters to file
# store parameters in dictionary
runparameters = {
    'operator type': flag2,
    '1:-div A grad phi = lambda phi, 2:-div A grad phi = lambda A phi, 3: -div grad phi = lambda A phi':"" ,
    'bctype': bctype,
    'deg': deg,
    'nelts': nelts,
    'npts': npts,
    'params': params,
    'coeftype': coeftype,
    'nc':nc,
    'x0': x0,
    'x1': x1,
    'a0': a0,
    'a1': a1,
    'shift':s,
    }
paramf = open(paramfile, 'w')
json.dump(runparameters, paramf, indent=4)
paramf.write('\n')
paramf.close()
PETSc.Sys.Print("> run parameters written to {}".format(paramfile))

mesh = IntervalMesh(nelts, x0, x1)
x , = SpatialCoordinate(mesh)
# define coefficient A
# pw constants alternately equal to a0 or a1
aelt = 'DG'
adeg = 0
aexpr=Function(FunctionSpace(IntervalMesh(nc, x0, x1),aelt,adeg))
#aval=a0+np.random.rand(nc2)*(a1-a0)
#aval=np.zeros(nc)
#for i in range(nc):
#   if i % 2 ==0:
#      aval[i]=a0
#   else:
#      aval[i]=a1
aval=a0+np.random.binomial(size=nc, n=1, p= 0.5)*(a1-a0)
aexpr.vector().set_local(aval)
V=FunctionSpace(mesh,aelt,adeg)
A = assemble(interpolate(aexpr, V))
# evaluate coefficient, save to file and plot
plt.clf()
print("> evaluating coefficient")
pts = np.linspace(x0, x1, npts, endpoint=True)
#avals = eval_u(A,pts)
avals=A.at(pts)
plt.plot(pts, avals, alpha=.75, linewidth=2)
plt.xlim([x0, x1])
plt.title('coefficient')
plt.savefig(coefplotfile, dpi=500)
PETSc.Sys.Print("> coefficient plotted to {}".format(coefplotfile))

# plot (shifted) landscape function 
uh=get_shifted_landscape(mesh,x0,x1,s,A,deg,npts,Landplotfile)
plt.clf()
pts = np.linspace(x0, x1, npts, endpoint=True)
avals = eval_u(uh,pts)
plt.plot(pts, avals, alpha=.75, linewidth=2)
plt.xlim([x0, x1])
plt.title('shifted landscape')
plt.savefig(Landplotfile, dpi=500)
PETSc.Sys.Print("> LL plotted to {}".format(Landplotfile))

# solve eigen problem and save results
EPS, nconv, Bsc, V=eigen_solver(mesh,A,deg,nreq,target,bctype,x0,x1,flag2)
modes, eigenvalues2, pratio = get_eigenpairs(mesh,EPS,nreq,Bsc,V,x0,x1,nelts,npts,plotefuns,plotefuns_2,eigenvalfile,eigenfunplotfile,eigenfunh5file,eigenfunmontagefile,eigenfunmontagefile_2,[],flag,eigenfunmon_all,ncol_plot)
np.savetxt(pratiofile,pratio)

plt.clf()
plt.scatter(modes,pratio)
plt.xlabel('modes')
plt.ylabel('p-ratio')
plt.savefig(mpfile)
PETSc.Sys.Print("> pratio vs modes to {}".format(mpfile))

plt.clf()
plt.scatter(eigenvalues2,pratio)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile)
PETSc.Sys.Print("> pratio vs eigenvalues to {}".format(epfile))


plt.clf()
plt.yscale('log')
plt.scatter(modes,pratio)
plt.xlabel('modes')
plt.ylabel('p-ratio')
plt.savefig(mpfile_log)
PETSc.Sys.Print("> pratio vs modes to {}".format(mpfile_log))

plt.clf()
plt.yscale('log')
plt.scatter(eigenvalues2,pratio)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_log)
PETSc.Sys.Print("> pratio vs eigenvalues to {}".format(epfile_log))
