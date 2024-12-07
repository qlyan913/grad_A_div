"""
Solve the eigenvalue problem with variable coefficient:
   -(Au')'=lambda u on square [0,L]x[0,L]
Here, we consider the random displacement model:
   A(x) = 1/(1+ sum_{integer n: x0<= n <= x1}f(x-n-dn(w))
  
   f1 = 20[max{(1-x^2/s^2)^3,0}(3x^2+1)], supp(f) in B_0(s)
   f2 =(-1)^(i+j) x 0.75 [max{(1-x^2/s^2)^3,0}(3x^2+1)], supp(f) in B_0(s)
   dn uniform distribution on the ball B_0(d_max)
   We choose s=1/4 and dmax=1/5 such that s+dmax<1/2
"""
import random
import os,math, json
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from slepc4py import SLEPc
import numpy as np
from solver import *
deg = 5
L=100 # length of square
nx=200
ny=200
nreq=301
target=20
plotefuns=0,10,20,30,40,50,60,70,80,90,100,150,200,250,300
plotefuns_2=[int(d) for d in range(20)]
flag=0 # 1: print all first n_all(default=500) eigenfuns, 0: print plotefuns 
f_flag=3 # 1: coef--- f1, 2: coef --- f2
n_all=300 
flag2 = 1
"""
     flag2 ---- 1: -div A grad phi = lambda phi
          ---- 2: -div A grad phi = lambda A phi
          ---- 3: -div grad phi = lambda A phi
"""
plotmesh=1 # 1: plot mesh. 0: no plot
bctype='dirichlet' # dirichlet or neumann
coeftype='random displacement' #'constant' #'random displacement' # 'fixed displacement' 
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
eigenfun_smpr_file= outdir + '/' + 'smpr_{:05d}.png'
eigenfunmontagefile = outdir + '/'+'eigenfunmontage.png'
eigenfunmontagefile_2 = outdir + '/'+'eigenfunmontage_v2.png'
eigenfunmon_all = outdir+'/'+'eigenfunmon{:03d}_{:03d}.png'
paramfile = outdir+ '/'+'Parameter.json'
signfile = outdir+ '/'+'sign_list.txt'

# write parameters to file
# store parameters in dictionary
runparameters = {
    'operator type': flag2,
    '1:-div A grad phi = lambda phi, 2:-div A grad phi = lambda A phi, 3: -div grad phi = lambda A phi':"" ,
    'bctype': bctype,
    'coef_type': f_flag,
    'deg': deg,
    'target':target,
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
print("> mesh with {} elements".format(mesh.num_cells()))
if plotmesh==1:
   plt.clf()
   triplot(mesh)
   plt.xlabel('X')
   plt.ylabel('Y')
   plt.title('Mesh')
   plt.savefig(meshplotfile, dpi=300)
   plt.close()
   print("> mesh plotted to {}".format(meshplotfile))

x , y= SpatialCoordinate(mesh)
# define coefficient A
if coeftype=='constant':
   aval=1
   aexpr = Constant(aval)
   aelt = 'DG'
   adeg = 0
   A = assemble(interpolate(aexpr, FunctionSpace(mesh, aelt, adeg)))
else: 
   s=0.25
   nn=L-1
   if coeftype == 'random displacement':
      # unifrom distribution on unit disk: https://rh8liuqy.github.io/Uniform_Disk.html
      U1 = np.random.uniform(size = nn**2)
      U2 = np.random.uniform(size = nn**2)
      r = dmax
      dn1 = r * np.sqrt(U2) * np.cos(2 * np.pi * U1)
      dn2 = r * np.sqrt(U2) * np.sin(2 * np.pi * U1)
      sampleplotfile = outdir+ '/'+'sample_dn.png'
      plt.clf()
      circle = plt.Circle((0, 0), r, color='g', fill=False)
      fig,ax = plt.subplots()
      ax.scatter(x = dn1, y = dn2, s = 0.5)
      ax.add_artist(circle)
      plt.title('sampling of dn')
      plt.savefig(sampleplotfile, dpi=300)
      print("> sampling of dn plotted to {}".format(sampleplotfile))
   elif coeftype == 'fixed displacement':
      dn1=np.zeros(nn**2)
      dn2=np.zeros(nn**2)
   Asum=Function(FunctionSpace(mesh, 'CG', 7))
   sign_list=[]
   for i in range(0,nn):
      for j in range(0,nn):
          x_center=[i+1+dn1[i*nn+j],j+1+dn2[i*nn+j]]
          if f_flag ==1:
             fi = conditional(((x-x_center[0])**2+(y-x_center[1])**2)**0.5>s,0,20*(1-((x-x_center[0])**2+(y-x_center[1])**2)/pow(s,2))**3*(3*((x-x_center[0])**2+(y-x_center[1])**2)+1))
          elif f_flag ==2:
            # sign=random.choice([-1,1])
             sign=(-1)**(i+j)
             fi = conditional(((x-x_center[0])**2+(y-x_center[1])**2)**0.5>s,0,sign*0.75*(1-((x-x_center[0])**2+(y-x_center[1])**2)/pow(s,2))**3*(3*((x-x_center[0])**2+(y-x_center[1])**2)+1))
          else:
             sign=(-1)**(i+j)
             if sign==-1:
                sign_2=-0.9
             else: 
                sign_2=20
             fi = conditional(((x-x_center[0])**2+(y-x_center[1])**2)**0.5>s,0,sign_2*(1-((x-x_center[0])**2+(y-x_center[1])**2)/pow(s,2))**3*(3*((x-x_center[0])**2+(y-x_center[1])**2)+1))
          Fi = assemble(interpolate(fi,FunctionSpace(mesh,'CG',7)))
          Asum += Fi
   A=assemble(interpolate(Constant(1.0)/(Constant(1.0)+Asum),FunctionSpace(mesh,'CG',7)))
#if f_flag==2:
 #  np.savetxt(signfile, sign_list)

# evaluate coefficient, save to file and plot
plt.clf()
fig, axes = plt.subplots()
collection = tripcolor(A, axes=axes)
fig.colorbar(collection);
plt.title('coefficient')
plt.savefig(coefplotfile, dpi=500)
print("> coefficient plotted to {}".format(coefplotfile))

# solve eigen problem and save results
EPS, nconv, Bsc, V=eigen_solver(mesh,A,deg,nreq,target,bctype,flag2)
modes, eigenvalues2, pratio = get_eigenpairs(EPS,nreq,Bsc,V,L,plotefuns,plotefuns_2,eigenvalfile,eigenfunplotfile,eigenfunmontagefile,eigenfunmontagefile_2,eigenfun_smpr_file,[],flag,eigenfunmon_all,n_all)
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
