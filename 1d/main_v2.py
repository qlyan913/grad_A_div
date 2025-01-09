"""
Solve the eigenvalue problem with variable coefficient:
   -(Au')'=lambda u on [x0,x1]
Here, we consider the 1d random displacement model:
   A(x) = 1/(1+ sum_{integer n: x0<= n <= x1}f(x-n-dn(w))
   f = 1/8[max{(1-x^2/s^2)^3,0}(3x^2+1)]', supp(f) in [-s,s]
   dn uniform distribution on [-dmax,dmax]
   We choose s=1/4 and dmax=1/5 such that s+dmax<1/2
"""
import os,math, json,csv
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from slepc4py import SLEPc
import numpy as np
from solver import *
deg = 5
L=200
nelts=10*L   # number of elements on interval
npts=2*nelts # for plotting functions
x0=0
x1=L
nc=200
nreq=300
target=0
plotefuns=0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600,700,800,900,999
plotefuns=[int(d) for d in range(180,220)]
plotefuns_2=[int(d) for d in range(20)]
fv=1 #fv=1: f, fv=2, f version 2
flag=0 # print all first nreq  eigenfuns
flag2 =2
"""
     flag2 ---- 1: -div A grad phi = lambda phi
          ---- 2: -div A grad phi = lambda A phi
          ---- 3: -div grad phi = lambda A phi
"""
bctype='dirichlet' # dirichlet or neumann
coeftype='random displacement' #'landscape' #'pw_2constant' #'random displacement' # 'fixed displacement' #'pw_2constant' #'random displacement'
dmax=0.2
a0=1
a1=10
np.random.seed(5)
#coeftype='constant'
params=''
# create directory and filenames for output
outdir = makedir()
# filenames
coefplotfile = outdir + '/' + 'coefficient.png'
eigenvalfile = outdir + '/' + 'eigenvalues.txt'
mpfile = outdir + '/' + 'pratio_mode.png'
epfile = outdir + '/' + 'pratio_eigen.png'
mpfile_log = outdir + '/' + 'pratio_mode_log.png'
epfile_log = outdir + '/' + 'pratio_eigen_log.png'
pratiofile=outdir + '/' + 'pratio.txt'
eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
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
    'x0': x0,
    'x1': x1,
    'a0': a0,
    'a1': a1,
    'nc':nc 
   }
paramf = open(paramfile, 'w')
json.dump(runparameters, paramf, indent=4)
paramf.write('\n')
paramf.close()
print("> run parameters written to {}".format(paramfile))

mesh = IntervalMesh(nelts, x0, x1)
x , = SpatialCoordinate(mesh)
# define coefficient A
if coeftype=='constant':
   a0=1
   a1=10
   aval=a0+np.random.rand()*(a1-a0)
   aexpr = Constant(aval)
   aelt = 'DG'
   adeg = 0
elif coeftype == 'pw_2constant':
   # pw constants alternately equal to a0 or a1
   center_list=range(nc)
   aval=a0+np.random.rand(nc)*(a1-a0)
   aelt = 'DG'
   adeg=0
   # create pw constant with nc pieces
   aexpr = Function(FunctionSpace(IntervalMesh(nc, x0, x1), aelt, adeg))
   aexpr.vector().set_local(aval)
elif coeftype == 'landscape':
   Vval=a0+np.random.rand(nc)*(a1-a0)
   Vexpr=Function(FunctionSpace(IntervalMesh(nc, x0, x1),'DG',0))
   Vexpr.vector().set_local(Vval)
   Vp=assemble(interpolate(Vexpr, FunctionSpace(mesh, 'DG', 0)))
   u=get_landscape(mesh,x0,x1,Vp,deg,npts,Vplotfile,Landplotfile)
   aexpr=u**2
   aelt = 'CG'
   adeg = 5
elif coeftype == '1/V^2':
   Vval=a0+np.random.rand(nc)*(a1-a0)
   Vexpr=Function(FunctionSpace(IntervalMesh(nc, x0, x1),'DG',0))
   Vexpr.vector().set_local(Vval)
   Vp=assemble(interpolate(Vexpr, FunctionSpace(mesh, 'DG', 0)))
   V_inv=1/(Vval**2)
   aelt = 'DG'
   adeg=0
   aexpr = Function(FunctionSpace(IntervalMesh(nc, x0, x1), aelt, adeg))
   aexpr.vector().set_local(V_inv)
   plt.clf()
   pts = np.linspace(x0, x1, npts, endpoint=True)
   Vpvals = eval_u(Vp,pts)
   plt.plot(pts, Vpvals, alpha=.75, linewidth=2)
   plt.xlim([x0, x1])
   plt.title('potential V')
   plt.savefig(Vplotfile, dpi=300)
   print("> potential V  plotted to {}".format(Vplotfile))
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
      if fv==1:
         f_sum=f_sum + conditional(abs(x-x_center)>s,0,6/8*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**3-6/(8*pow(s,2))*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**2*(3*pow(x-x_center,2)+1))
      else:   
         f_sum=f_sum + conditional(abs(x-x_center)>s,0,20*(1-(x-x_center)**2/pow(s,2))**3*(3*(x-x_center)**2+1))
   
   if fv==1:
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
plt.title('coefficient')
plt.savefig(coefplotfile, dpi=500)
print("> coefficient plotted to {}".format(coefplotfile))

# solve eigen problem and save results
EPS, nconv, Bsc, V=eigen_solver(mesh,A,deg,nreq,target,bctype,x0,x1,flag2)
modes, eigenvalues2, pratio =get_eigenpairs(mesh,EPS,nreq,Bsc,V,x0,x1,nelts,npts,plotefuns,plotefuns_2,eigenvalfile,eigenfunplotfile,eigenfunh5file,eigenfunmontagefile,eigenfunmontagefile_2,[],flag,eigenfunmon_all,5)
np.savetxt(pratiofile,pratio)
eigenvalues_list=[]
pratio_list=[]
eigenvalues_list+=eigenvalues2
pratio_list+=pratio
with open(eigen_pratiofile, 'w', newline='') as csvfile:
    fieldnames = ['eigenvalue','participation_ratio']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(eigenvalues_list)):
       writer.writerow({'eigenvalue':eigenvalues_list[i],'participation_ratio':pratio_list[i]})
PETSc.Sys.Print("> Results of eigenvalues and participation ratio  are saved to {}".format(eigen_pratiofile))

plt.clf()
plt.scatter(modes,pratio,s=0.5)
plt.xlabel('modes')
plt.ylabel('p-ratio')
plt.savefig(mpfile,dpi=300)
print("> pratio vs modes to {}".format(mpfile))

plt.clf()
plt.scatter(eigenvalues2,pratio,s=0.5)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile,dpi=300)
print("> pratio vs eigenvalues to {}".format(epfile))


plt.clf()
plt.yscale('log')
plt.scatter(modes,pratio,s=0.5)
plt.xlabel('modes')
plt.ylabel('p-ratio')
plt.savefig(mpfile_log,dpi=300)
print("> pratio vs modes to {}".format(mpfile_log))

plt.clf()
plt.yscale('log')
plt.scatter(eigenvalues2,pratio,s=0.5)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_log,dpi=300)
print("> pratio vs eigenvalues to {}".format(epfile_log))

