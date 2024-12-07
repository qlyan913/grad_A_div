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
sigma_1=0.5
plotefun_flag=0 # 0 -- solve eigenproblem only wihtout plotting
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
outdir = 'Results'
# filenames
coefh5file = outdir + '/h5_file/' + 'coef.h5'
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

if plotmesh==1:
   plt.clf()
   triplot(mesh)
   plt.xlabel('X')
   plt.ylabel('Y')
   plt.title('Mesh')
   plt.savefig(meshplotfile, dpi=300)
   plt.close()
   print("> mesh plotted to {}".format(meshplotfile))

# read mesh and coefficient file
with CheckpointFile(coefh5file,'r') as afile:
     mesh=afile.load_mesh()
     A=afile.load_function(mesh,"coef")

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
