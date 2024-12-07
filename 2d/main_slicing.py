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
deg = 5
L=100 # length of square
nx=L
ny=L
nc=5
nc2=nc*nc
a0=1   # pc constant range from [a0, a1]
a1=10 
sigma_0=0
sigma_1=1
plotefun_flag=1 # 0 -- solve eigenproblem only wihtout plotting
flag2=2
"""
     flag2 ---- 1: -div A grad phi = lambda phi
          ---- 2: -div A grad phi = lambda A phi
          ---- 3: -div grad phi = lambda A phi
"""
plotmesh=0 # 1: plot mesh. 0: no plot
bctype='dirichlet' # dirichlet or neumann
coeftype='pc constant'
np.random.seed(5)
#coeftype='constant'
params=''
# create directory and filenames for output
outdir = makedir()
#outdir = 'Results/000022'
# filenames
coefplotfile = outdir + '/' + 'coefficient.png'
meshplotfile = outdir + '/' + 'mesh.png'
epfile_log = outdir + '/' + 'pratio_eigen_log.png'
epfile_loglog = outdir + '/' + 'pratio_eigen_loglog.png'
eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
eigenfunplotfile = outdir + '/' + 'eigen{:05d}.png'
eigenfunh5file  = outdir + '/h5_file/' + 'eigenfun{:05d}.h5'
eigenfun_smpr_file= outdir + '/' + 'smpr_{:05d}.png'
eigenfunmontagefile = outdir + '/'+'eigenfunmontage_mode_{:03d}.png'
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
    'nx': nx,
    'ny':ny,
    'a0':a0,
    'a1':a1,
    'params': params,
    'coeftype': coeftype,
    'L':L,
    'nc':nc2,
    'sigma_0':sigma_0,
    'sigma_1':sigma_1
    }
paramf = open(paramfile, 'w')
json.dump(runparameters, paramf, indent=4)
paramf.write('\n')
paramf.close()
PETSc.Sys.Print("> run parameters written to {}".format(paramfile))

mesh = SquareMesh(nx,ny,L,quadrilateral=True)
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
aval=a0+np.random.rand(nc2)*(a1-a0)
aexpr=Function(FunctionSpace(SquareMesh(nc,nc,L,quadrilateral=True),aelt,adeg))
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

# solve eigen problem and save results
PETSc.Sys.Print("> solving for interval from {} to {}".format(sigma_0,sigma_1))
EPS, nconv, Bsc, V=eigen_solver_slicing(mesh,A,deg,sigma_0,sigma_1,bctype,flag2)
plotefuns=[int(d) for d in range(nconv)]
if plotefun_flag == 1 :
   modes, eigenvalues2, pratio,eigenf_imgs_smpr = get_eigenpairs_v3(EPS,nconv,Bsc,V,L,plotefuns,eigenfunplotfile,eigenfunmontagefile,eigenfun_smpr_file,eigenfunh5file,mesh)
   eigenvalues_list+=eigenvalues2
   pratio_list+=pratio
   eigf_imgs_list+=eigenf_imgs_smpr

with open(eigen_pratiofile, 'w', newline='') as csvfile:
    fieldnames = ['eigenvalue','participation_ratio']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(eigenvalues_list)):
       writer.writerow({'eigenvalue':eigenvalues_list[i],'participation_ratio':pratio_list[i]})
PETSc.Sys.Print("> Results of eigenvalues and participation ratio  are saved to {}".format(eigen_pratiofile))

plt.clf()
plt.yscale('log')
plt.scatter(eigenvalues_list,pratio_list)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_log)
PETSc.Sys.Print("> pratio vs eigenvalues to {}".format(epfile_log))

plt.clf()
plt.yscale('log')
plt.xscale('log')
plt.scatter(eigenvalues_list,pratio_list)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_loglog)
PETSc.Sys.Print("> pratio vs eigenvalues to {}".format(epfile_loglog))

plt.clf()
plt.scatter(modes,pratio)
plt.xlabel('modes')
plt.ylabel('p-ratio')
plt.savefig(mpfile)
PETSc.Sys.Print("> pratio vs modes to {}".format(mpfile))

plt.clf()
plt.yscale('log')
plt.scatter(modes,pratio)
plt.xlabel('modes')
plt.ylabel('p-ratio')
plt.savefig(mpfile_log)
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


