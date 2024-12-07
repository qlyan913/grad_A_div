"""
Solve the eigenvalue problem with variable coefficient:
   -div(A\nabla u')=lambda u on square [0,L]x[0,L]x[0,L]
Here, A(x)  is piecewise constant on hexahedral mesh,taking unfirom random number from [a01, a1]
"""
#from mpi4py import MPI
import csv
import random
import os,math, json
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
#from mpi4py import MPI
from firedrake.__future__ import interpolate
from slepc4py import SLEPc
import numpy as np
from solver import *
deg = 3
L=100 # length of square
nx=L
ny=L
nz=L
a0=1   # pc constant range from [a0, a1]
a1=10  
nreq=4
#target_list=[0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600,800,1000,1500,2000,2500,3000,3500,4000]
target_list=[0,10,20,40,60,80,100,200,250,300,400,500,600,800,1000,1500,2000]
#target_list =[ 0]
plotefuns=[int(d) for d in range(nreq)]
solver_flag=2
flag2=1
"""
     flag2 ---- 1: -div A grad phi = lambda phi
          ---- 2: -div A grad phi = lambda A phi
          ---- 3: -div grad phi = lambda A phi
"""
plotmesh=1 # 1: plot mesh. 0: no plot
bctype='dirichlet' # dirichlet or neumann
coeftype='pc constant'
np.random.seed(5)
#coeftype='constant'
params=''
# create directory and filenames for output
outdir ='Results/000015' # makedir()
# filenames
coefplotfile = outdir + '/' + 'coefficient.pvd'
meshplotfile = outdir + '/' + 'mesh.pvd'
eigen_pratiofile=outdir + '/' + 'eigen_pratio.csv'
epfile_log = outdir + '/' + 'pratio_eigen_log.png'
epfile_loglog = outdir + '/' + 'pratio_eigen_loglog.png'
eigenfunplotfile = outdir + '/' + 'target_{:05d}_eigen.pvd'
eigenfun_smpr_file= outdir + '/' + 'target_{:05d}_smpr.pvd'
paramfile = outdir+ '/'+'Parameter.json'
signfile = outdir+ '/'+'sign_list.txt'

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
    'nz':nz,
    'a0':a0,
    'a1':a1,
    'params': params,
    'coeftype': coeftype,
    'L':L,
    'eigen_solver': solver_flag,
    'nreq': nreq
    }
paramf = open(paramfile, 'w')
json.dump(runparameters, paramf, indent=4)
paramf.write('\n')
paramf.close()
PETSc.Sys.Print("> run parameters written to {}".format(paramfile))

mesh = CubeMesh(nx,ny,nz,L,hexahedral=True)
nc=mesh.num_cells()
#print("common world rank", COMM_WORLD.rank, " mesh with {} elements".format(nc) )
outfile = File(meshplotfile)
outfile.write(mesh)
PETSc.Sys.Print("> File for visualization in Paraview saved to {}".format(meshplotfile))

# define coefficient A
aelt = 'DG'
adeg = 0
V=FunctionSpace(mesh, aelt, adeg)
aexpr = Function(V)
#print("common world rank", COMM_WORLD.rank, "number of elements", mesh.num_cells())
aval=a0+np.random.rand(nc)*(a1-a0)
aexpr.vector().set_local(aval)
A = assemble(interpolate(aexpr,V)) # FunctionSpace(mesh, aelt, adeg)))
A.rename("coeff")
# evaluate coefficient, save to file and plot
outfile=File(coefplotfile)
outfile.write(A)
PETSc.Sys.Print("> coefficient plotted to {}".format(coefplotfile))

eigenvalues_list=[]
pratio_list=[]
eigf_imgs_list=[]
targets_all=[]
# solve eigen problem and save results
for target in target_list: 
   PETSc.Sys.Print("solving for target: ",target) 
   EPS, nconv, Bsc, V=eigen_solver(mesh,A,deg,nreq,target,bctype,solver_flag,flag2)
   modes, eigenvalues2, pratio,targets = get_eigenpairs_v2(EPS,nreq,Bsc,V,L,plotefuns,eigenfunplotfile,eigenfun_smpr_file,target)
   del EPS, Bsc, V
   eigenvalues_list+=eigenvalues2
   pratio_list+=pratio
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
PETSc.Sys.Print("> pratio vs eigenvalues to {}".format(epfile_log))

plt.clf()
plt.yscale('log')
plt.xscale('log')
plt.scatter(eigenvalues_list,pratio_list)
plt.xlabel('eigenvalues')
plt.ylabel('p-ratio')
plt.savefig(epfile_loglog)
PETSc.Sys.Print("> pratio vs eigenvalues to {}".format(epfile_loglog))
