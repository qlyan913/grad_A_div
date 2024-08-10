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
deg = 5
L=1 # length of square
nx=100
ny=100
nreq=1000
target=0
plotefuns=0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600,700,800,900,999
plotefuns_2=[int(d) for d in range(20)]
flag=0 # 1: print all first 500 eigenfuns, 0: print plotefuns 
flag2 =1
"""
     flag2 ---- 1: -div A grad phi = lambda phi
          ---- 2: -div A grad phi = lambda A phi
          ---- 3: -div grad phi = lambda A phi
"""
bctype='dirichlet' # dirichlet or neumann
coeftype='constant' #'random displacement' # 'fixed displacement' #'pw_2constant' #'random displacement'
dmax=0.2
np.random.seed(5)
#coeftype='constant'
params=''
# create directory and filenames for output
outdir = makedir()
# filenames
coefplotfile = outdir + '/' + 'coefficient.png'
meshplotfile = outdir + '/' + 'mesh.png'
eigenvalfile = outdir + '/' + 'eigenvalues.txt'
mpfile = outdir + '/' + 'pratio_mode.png'
epfile = outdir + '/' + 'pratio_eigen.png'
mpfile_log = outdir + '/' + 'pratio_mode_log.png'
epfile_log = outdir + '/' + 'pratio_eigen_log.png'
pratiofile=outdir + '/' + 'pratio.txt'
eigenfunplotfile = outdir + '/' + 'eigenfun{:05d}.png'
eigenfunmontagefile = outdir + '/'+'eigenfunmontage.png'
eigenfunmontagefile_2 = outdir + '/'+'eigenfunmontage_v2.png'
eigenfunmon_all = outdir+'/'+'eigenfunmon{:03d}_{:03d}.png'
paramfile = outdir+ '/'+'Parameter.json'

# write parameters to file
# store parameters in dictionary
runparameters = {
    'operator type': flag2,
    '1:-div A grad phi = lambda phi, 2:-div A grad phi = lambda A phi, 3: -div grad phi = lambda A phi':"" ,
    'bctype': bctype,
    'deg': deg,
    'nx': nx,
    'ny':ny,
    'params': params,
    'coeftype': coeftype,
    'L':L
    }
paramf = open(paramfile, 'w')
json.dump(runparameters, paramf, indent=4)
paramf.write('\n')
paramf.close()
print("> run parameters written to {}".format(paramfile))

mesh = SquareMesh(nx,ny,L)
plt.clf()
triplot(mesh)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mesh')
plt.savefig(meshplotfile, dpi=300)
print("> mesh plotted to {}".format(meshplotfile))

x , y= SpatialCoordinate(mesh)
# define coefficient A
if coeftype=='constant':
   aval=1
   aexpr = Constant(aval)
   aelt = 'DG'
   adeg = 0
elif coeftype == 'pw_2constant':
   # pw constants alternately equal to a0 or a1
   nc=x1-x0
   a0=1
   a1=10
   aval=np.zeros(nc)
   center_list=range(nc)
   for i in range(nc):
       if i % 2 ==0:
          aval[i]=a0
       else:
          aval[i]=a1
   aelt = 'DG'
   adeg=0
   # create pw constant with nc pieces
   aexpr = Function(FunctionSpace(IntervalMesh(nc, x0, x1), aelt, adeg))
   aexpr.vector().set_local(aval)
else: 
   s=0.25
   nn=x1-x0-1
   if coeftype == 'random displacement':
      dn=-dmax+np.random.rand(nn+1)*(2*dmax)
   elif coeftype == 'fixed displacement':
      dn=np.zeros(nn+1)
   f_sum=0.0
   for i in range(nn):
      x_center=i+1+dn[i]
      f_sum=f_sum + conditional(abs(x-x_center)>s,0,6/8*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**3-6/(8*pow(s,2))*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**2*(3*pow(x-x_center,2)+1))
   # near x=x0
   x_center=x0+dn[nn]
   f_sum=f_sum + conditional(abs(x-x_center)>s,0,6/8*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**3-6/(8*pow(s,2))*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**2*(3*pow(x-x_center,2)+1))
   # near x=x1
   x_center=x_center+x1-x0
   f_sum=f_sum + conditional(abs(x-x_center)>s,0,6/8*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**3-6/(8*pow(s,2))*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**2*(3*pow(x-x_center,2)+1))
   aexpr=1./(1+f_sum)
   aelt='CG'
   adeg=7

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
EPS, nconv, Bsc, V=eigen_solver(mesh,A,deg,nreq,target,bctype,x0,x1,flag2)
modes, eigenvalues2, pratio = get_eigenpairs(EPS,nreq,Bsc,V,x0,x1,nelts,npts,plotefuns,plotefuns_2,eigenvalfile,eigenfunplotfile,eigenfunmontagefile,eigenfunmontagefile_2,[],flag,eigenfunmon_all)
np.savetxt(pratiofile,pratio)

plt.clf()
plt.scatter(modes,pratio)
plt.xlabel('modes')
plt.ylabel('p-ratio')
plt.savefig(mpfile)
print("> pratio vs modes to {}".format(mpfile))

plt.clf()
plt.scatter(eigenvalues2,pratio)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile)
print("> pratio vs eigenvalues to {}".format(epfile))


plt.clf()
plt.yscale('log')
plt.scatter(modes,pratio)
plt.xlabel('modes')
plt.ylabel('p-ratio')
plt.savefig(mpfile_log)
print("> pratio vs modes to {}".format(mpfile_log))

plt.clf()
plt.yscale('log')
plt.scatter(eigenvalues2,pratio)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_log)
print("> pratio vs eigenvalues to {}".format(epfile_log))
