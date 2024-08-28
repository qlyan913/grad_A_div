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
import csv
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
nreq=21
target_list=[0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600,800,1000,1200,1500,2000]
plotefuns=[int(d) for d in range(20)]
f_flag=1 # 1: coef--- f1, 2: coef --- f2
flag2 =3
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
epfile_log = outdir + '/' + 'pratio_eigen_log.png'
epfile_loglog = outdir + '/' + 'pratio_eigen_loglog.png'
eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
eigenfunplotfile = outdir + '/' + 'target_{:05d}_eigen{:05d}.png'
eigenfun_smpr_file= outdir + '/' + 'target_{:05d}_smpr_{:05d}.png'
eigenfunmontagefile = outdir + '/'+'eigenfunmontage_target_{:05d}.png'
eigenfunmontagefile_smpr=outdir+'/'+'eigenfunmontage_smpr.png'
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
    'target':target_list,
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

# evaluate coefficient, save to file and plot
plt.clf()
fig, axes = plt.subplots()
collection = tripcolor(A, axes=axes)
fig.colorbar(collection);
plt.title('coefficient')
plt.savefig(coefplotfile, dpi=500)
print("> coefficient plotted to {}".format(coefplotfile))

eigenvalues_list=[]
pratio_list=[]
eigf_imgs_list=[]
targets_all=[]
# solve eigen problem and save results
for target in target_list: 
   EPS, nconv, Bsc, V=eigen_solver(mesh,A,deg,nreq,target,bctype,flag2)
   modes, eigenvalues2, pratio,eigenf_imgs_smpr,targets= get_eigenpairs_v2(EPS,nreq,Bsc,V,L,plotefuns,eigenfunplotfile,eigenfunmontagefile,eigenfun_smpr_file,target)
   eigenvalues_list+=eigenvalues2
   pratio_list+=pratio
   eigf_imgs_list+=eigenf_imgs_smpr
   targets_all+=targets

with open(eigen_pratiofile, 'w', newline='') as csvfile:
    fieldnames = ['target','eigenvalue','participation_ratio']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(eigenvalues_list)):
       writer.writerow({'target':targets_all[i],'eigenvalue':eigenvalues_list[i],'participation_ratio':pratio_list[i]})
print("> Results of eigenvalues and participation ratio  are saved to {}".format(eigen_pratiofile))

plt.clf()
plt.yscale('log')
plt.scatter(eigenvalues_list,pratio_list)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_log)
print("> pratio vs eigenvalues to {}".format(epfile_log))

plt.clf()
plt.yscale('log')
plt.xscale('log')
plt.scatter(eigenvalues_list,pratio_list)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_loglog)
print("> pratio vs eigenvalues to {}".format(epfile_loglog))

n_imgs=len(eigf_imgs_list)
if n_imgs<25:
   combine_images(columns=5, space=20, images=eigf_imgs_list,file=eigenfunmontagefile_smpr)
   print("> eigenfunction montage written to {}".format(eigenfunmontagefile_smpr.format(0)))
else:
   dd,dd2=divmod(n_imgs,25)
   for i in range(dd):
       segment=list(range(25*i,25*i+25))
       i0=segment[0]
       iend=segment[-1]
       eigenf_imgs=[]
       for j in segment:
          eigenf_imgs.append(eigf_imgs_list[j])
       combine_images(columns=5, space=20, images=eigenf_imgs,file=eigenfunmontagefile_smpr.format(i0))
       print("> eigenfunction montage  written to {}".format(eigenfunmontagefile_smpr.format(i0)))
   segment=list(range(25*dd,n_imgs))
   i0=segment[0]
   eigenf_imgs=[]
   for j in segment:
      eigenf_imgs.append(eigf_imgs_list[j])
   combine_images(columns=5, space=20, images=eigenf_imgs,file=eigenfunmontagefile_smpr.format(i0))
   print("> eigenfunction montage  written to {}".format(eigenfunmontagefile_smpr.format(i0)))
