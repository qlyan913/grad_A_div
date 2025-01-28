"""
COMM_WORLD.rankSolve the eigenvalue problem with variable coefficient:
   -div(A\nabla u')=lambda u on square [0,L]x[0,L]
Here, we consider the 1d random displacement model:
   A(x)  is piecewise constant on quadrilateral mesh, 
   taking unfirom random number from [a01, a1]
"""
import random
import csv
import os,math, json
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from slepc4py import SLEPc
import numpy as np
from solver import *
deg = 8
Lx=10 # length of square
Ly=10.1
nx=10*Lx
ny=nx
nc=2
nc2=nc**2
a0=1   # pc constant range from [a0, a1]
a1=10
nreq=101
#target_list=np.linspace(600,1000,401)
#target_list=[m/4 for m in range(41)]
target_list=[0]
plotefuns=[int(d) for d in range(nreq)]
flag2=2
"""
     flag2 ---- 1: -div A grad phi = lambda phi
          ---- 2: -div A grad phi = lambda A phi
          ---- 3: -div grad phi = lambda A phi
"""
plotmesh=0 # 1: plot mesh. 0: no plot
bctype='dirichlet' # dirichlet or neumann
coeftype='pc constant' #'checkerboard' #'pc constant'
np.random.seed(5)
#coeftype='constant'
params=''
# create directory and filenames for output
outdir = makedir()
#outdir = 'Results/000017'
# filenames
coefplotfile = outdir + '/' + 'coefficient.png'
meshplotfile = outdir + '/' + 'mesh.png'
epfile_log = outdir + '/' + 'pratio_eigen_log.png'
epfile_loglog = outdir + '/' + 'pratio_eigen_loglog.png'
eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
eigenfunh5file = outdir + '/h5_file/' + 'target_{:4.2f}_eigen{:05d}.h5'
eigenfunplotfile = outdir + '/' + 'target_{:4.2f}_eigen{:05d}.png'
eigenfun_smpr_file= outdir + '/' + 'target_{:4.2f}_smpr_{:05d}.png'
eigenfunmontagefile = outdir + '/'+'eigenfunmontage_target_{:4.2f}_mode_{:03d}.png'
eigenfunmontagefile_smpr=outdir+'/'+'eigenfunmontage_smpr_{:05d}.png'
paramfile = outdir+ '/'+'Parameter.json'
signfile = outdir+ '/'+'sign_list.txt'
mpfile = outdir + '/' + 'pratio_mode.png'
mpfile_log = outdir + '/' + 'pratio_mode_log.png'
avalfile=outdir+'/'+'aval.txt'
# write parameters to file
# store parameters in dictionary
runparameters = {
    'operator type': flag2,
    '1:-div A grad phi = lambda phi, 2:-div A grad phi = lambda A phi, 3: -div grad phi = lambda A phi':"" ,
    'bctype': bctype,
    'deg': deg,
 #   'target':target_list,
    'nx': nx,
    'ny':ny,
    'a0':a0,
    'a1':a1,
    'params': params,
    'coeftype': coeftype,
    'Lx':Lx,
    'Ly':Ly,
    'nc':nc2,
    'nreq':nreq
    }
paramf = open(paramfile, 'w')
json.dump(runparameters, paramf, indent=4)
paramf.write('\n')
paramf.close()
PETSc.Sys.Print("> run parameters written to {}".format(paramfile))

mesh = RectangleMesh(nx,ny,Lx,Ly,quadrilateral=True)
x , y= SpatialCoordinate(mesh)
#mesh2 = SquareMesh(nc,nc,L,quadrilateral=True)
#PETSc.Sys.Print("common world rank", COMM_WORLD.rank, " mesh with {} elements".format(nc2) )
#PETSc.Sys.Print("> mesh with {} elements".format(mesh.num_cells()))
if plotmesh==1:
   plt.clf()
   triplot(mesh)
   plt.xlabel('X')
   plt.ylabel('Y')
   plt.title('Mesh')
   plt.savefig(meshplotfile, dpi=300)
   plt.close()
   print("> mesh plotted to {}".format(meshplotfile))


# define coefficient A
aelt = 'DG'
adeg = 0
V=FunctionSpace(mesh,aelt,adeg)
if coeftype=='checkerboard':
   A=Function(V)
   wsize=L/nc
   for i in range(nc):
      if i%2==0:
         for j in range(nc):
            x_center=wsize*(i+0.5)
            y_center=wsize*(j+0.5)
            if j%2==0:
               fi=conditional(And(abs(x-x_center)<wsize*0.5,abs(y-y_center)<wsize*0.5),a0,0)
            else:
               fi=conditional(And(abs(x-x_center)<wsize*0.5,abs(y-y_center)<wsize*0.5),a1,0)
            Fi = assemble(interpolate(fi,V))
            A += Fi
      else:
         for j in range(nc):
            x_center=wsize*(i+0.5)
            y_center=wsize*(j+0.5)
            if j%2==1:
               fi=conditional(And(abs(x-x_center)<wsize*0.5,abs(y-y_center)<wsize*0.5),a0,0)
            else:
               fi=conditional(And(abs(x-x_center)<wsize*0.5,abs(y-y_center)<wsize*0.5),a1,0)
            Fi = assemble(interpolate(fi,V))
            A += Fi
else:
   #aval=a0+np.random.rand(nc2)*(a1-a0)
   #aval=a0+np.random.binomial(size=nc2, n=1, p= 0.5)*(a1-a0)
   aval=np.array([1,10,10,1])
#   aval=np.array([1,1,10,10])
   aexpr=Function(FunctionSpace(RectangleMesh(nc,nc,Lx,Ly,quadrilateral=True),aelt,adeg))
   aexpr.vector().set_local(aval)
   A = assemble(interpolate(aexpr, V))

# evaluate coefficient, save to file and plot
plt.clf()
fig, axes = plt.subplots()
collection = tripcolor(A, axes=axes)
fig.colorbar(collection);
plt.title('coefficient')
plt.savefig(coefplotfile, dpi=500)
PETSc.Sys.Print("> coefficient plotted to {}".format(coefplotfile))

eigenvalues_list=[]
pratio_list=[]
eigf_imgs_list=[]
targets_all=[]
# solve eigen problem and save results
for target in target_list: 
   PETSc.Sys.Print("> solving for target {}".format(target))
   EPS, nconv, Bsc, V=eigen_solver(mesh,A,deg,nreq,target,bctype,flag2)
   modes, eigenvalues2, pratio,eigenf_imgs_smpr,targets = get_eigenpairs_v2(EPS,nreq,Bsc,V,Lx,plotefuns,eigenfunplotfile,eigenfunmontagefile,eigenfun_smpr_file,target,mesh,eigenfunh5file)
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
PETSc.Sys.Print("> Results of eigenvalues and participation ratio  are saved to {}".format(eigen_pratiofile))

plt.clf()
plt.yscale('log')
plt.scatter(eigenvalues_list,pratio_list,s=0.5)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_log, dpi=300)
PETSc.Sys.Print("> pratio vs eigenvalues to {}".format(epfile_log))

plt.clf()
plt.yscale('log')
plt.xscale('log')
plt.scatter(eigenvalues_list,pratio_list,s=0.5)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_loglog,dpi=300)
PETSc.Sys.Print("> pratio vs eigenvalues to {}".format(epfile_loglog))

plt.clf()
plt.scatter(modes,pratio,s=0.5)
plt.xlabel('modes')
plt.ylabel('p-ratio')
plt.savefig(mpfile,dpi=300)
PETSc.Sys.Print("> pratio vs modes to {}".format(mpfile))

plt.clf()
plt.yscale('log')
plt.scatter(modes,pratio,s=0.5)
plt.xlabel('modes')
plt.ylabel('p-ratio')
plt.savefig(mpfile_log, dpi=300)
PETSc.Sys.Print("> pratio vs modes to {}".format(mpfile_log))

n_imgs=len(eigf_imgs_list)
if n_imgs<25:
   if n_imgs>0:
      combine_images(columns=5, space=20, images=eigf_imgs_list,file=eigenfunmontagefile_smpr)
      PETSc.Sys.Print("> eigenfunction montage written to {}".format(eigenfunmontagefile_smpr.format(0)))
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
       PETSc.Sys.Print("> eigenfunction montage  written to {}".format(eigenfunmontagefile_smpr.format(i0)))
   segment=list(range(25*dd,n_imgs))
   i0=segment[0]
   eigenf_imgs=[]
   for j in segment:
      eigenf_imgs.append(eigf_imgs_list[j])
   combine_images(columns=5, space=20, images=eigenf_imgs,file=eigenfunmontagefile_smpr.format(i0))
   PETSc.Sys.Print("> eigenfunction montage  written to {}".format(eigenfunmontagefile_smpr.format(i0)))


