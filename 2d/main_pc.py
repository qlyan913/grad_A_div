"""
Solve the eigenvalue problem with variable coefficient:
   -div(A\nabla u')=lambda u on square [0,L]x[0,L]
Here, we consider the 1d random displacement model:
   A(x)  is piecewise constant on quadrilateral mesh, 
   taking unfirom random number from [a01, a1]
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
L=10 # length of square
nx=L
ny=L
a0=1   # pc constant range from [a0, a1]
a1=10  
nreq=21
target_list=[0,5,10,15,20,25,30,35,40,45,50,55,60,70,80,90,100,150,200,250,300]
plotefuns=[int(d) for d in range(20)]
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
outdir = makedir()
# filenames
coefplotfile = outdir + '/' + 'coefficient.png'
meshplotfile = outdir + '/' + 'mesh.png'
eigenvalfile = outdir + '/' + 'eigenvalues_target_{:05d}.txt'
epfile_log = outdir + '/' + 'pratio_eigen_log.png'
epfile_loglog = outdir + '/' + 'pratio_eigen_loglog.png'
pratiofile=outdir + '/' + 'pratio_target_{:05d}.txt'
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
    'deg': deg,
    'target':target_list,
    'nx': nx,
    'ny':ny,
    'a0':a0,
    'a1':a1,
    'params': params,
    'coeftype': coeftype,
    'L':L
    }
paramf = open(paramfile, 'w')
json.dump(runparameters, paramf, indent=4)
paramf.write('\n')
paramf.close()
print("> run parameters written to {}".format(paramfile))

mesh = SquareMesh(nx,ny,L,quadrilateral=True)
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
nc=nx*ny
aval=a0+np.random.rand(nc)*(a1-a0)
aelt = 'DG'
adeg = 0
aexpr = Function(FunctionSpace(SquareMesh(nx,ny,L,quadrilateral=True),aelt,adeg))
aexpr.vector().set_local(aval)
A = assemble(interpolate(aexpr, FunctionSpace(mesh, aelt, adeg)))

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
# solve eigen problem and save results
for target in target_list: 
   EPS, nconv, Bsc, V=eigen_solver(mesh,A,deg,nreq,target,bctype,flag2)
   modes, eigenvalues2, pratio,eigenf_imgs_smpr = get_eigenpairs_v2(EPS,nreq,Bsc,V,L,plotefuns,eigenvalfile,eigenfunplotfile,eigenfunmontagefile,eigenfun_smpr_file,target)
   np.savetxt(pratiofile.format(target),pratio)
   eigenvalues_list+=eigenvalues2
   pratio_list+=pratio
   eigf_imgs_list+=eigenf_imgs_smpr

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

combine_images(columns=5, space=20, images=eigf_imgs_list,file=eigenfunmontagefile_smpr)
print("> eigenfunction montage written to {}".format(eigenfunmontagefile_smpr)) 
