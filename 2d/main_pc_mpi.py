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
from solver_mpi import *
deg = 5
L=10 # length of square
nx=L
ny=L
nc=2
nc2=nc*nc
a0=1   # pc constant range from [a0, a1]
a1=10 
nreq=51
#target_list=[0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600,800,1000,1200,1500,2000,2200,2500,3000]
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
coeftype='pc constant'
np.random.seed(5)
#coeftype='constant'
params=''
# create directory and filenames for output
#outdir = makedir()
outdir = 'Results/000040'
# filenames
coefplotfile = outdir + '/h5_file/' + 'coefficient.h5'
meshplotfile = outdir + '/' + 'mesh.png'
epfile_log = outdir + '/' + 'pratio_eigen_log.png'
epfile_loglog = outdir + '/' + 'pratio_eigen_loglog.png'
eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
eigenfunplotfile = outdir + '/h5_file/' + 'target_{:05d}_eigen{:05d}.h5'
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
    'target':target_list,
    'nx': nx,
    'ny':ny,
    'a0':a0,
    'a1':a1,
    'params': params,
    'coeftype': coeftype,
    'L':L,
    'nc':nc2,
    'nreq':nreq
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
mesh2 = SquareMesh(nc,nc,L,quadrilateral=True)
V2=FunctionSpace(mesh2,aelt,adeg)
aexpr = Function(V2)
av=aexpr.vector()
aval=a0+np.random.rand(nc2)*(a1-a0)
###
n_min,n_max=av.local_range()
aexpr.vector().set_local(aval[n_min:n_max])
A = Function(V, name="coef")
A.assign(assemble(interpolate(aexpr, V)))

# save coefficients
with CheckpointFile(coefplotfile,'w') as afile:
     afile.save_mesh(mesh)
     afile.save_function(A)
PETSc.Sys.Print("> coefficient saved to {}".format(coefplotfile))
modes_list=[]
eigenvalues_list=[]
pratio_list=[]
eigf_imgs_list=[]
targets_all=[]
# solve eigen problem and save results
for target in target_list: 
   PETSc.Sys.Print("> solving for target {}".format(target))
   EPS, nconv, Bsc, V=eigen_solver(mesh,A,deg,nreq,target,bctype,flag2)
   modes, eigenvalues2, pratio,targets = get_eigenpairs_v2(mesh,EPS,nreq,Bsc,V,L,plotefuns,eigenfunplotfile,target)
   eigenvalues_list+=eigenvalues2
   pratio_list+=pratio
   targets_all+=targets
   modes_list+=modes

with open(eigen_pratiofile, 'w', newline='') as csvfile:
    fieldnames = ['target','modes','eigenvalue','participation_ratio']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(eigenvalues_list)):
       writer.writerow({'target':targets_all[i],'modes':modes_list[i],'eigenvalue':eigenvalues_list[i],'participation_ratio':pratio_list[i]})
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


