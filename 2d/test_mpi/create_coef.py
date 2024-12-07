import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
import numpy as np
L=100
nx=L
ny=L
nc=5
nc2=nc**2
a0=1   # pc constant range from [a0, a1]
a1=10 
outdir = 'Results'
coefh5file= outdir + '/h5_file/'+'coef.h5'
#coefplotfile = outdir + '/' + 'coefficient.png'
mesh = SquareMesh(nx,ny,L,quadrilateral=True)
np.random.seed(5)
# define coefficient A
aelt = 'DG'
adeg = 0
V=FunctionSpace(mesh,aelt,adeg)
aval=a0+np.random.rand(nc2)*(a1-a0)
aexpr=Function(FunctionSpace(SquareMesh(nc,nc,L,quadrilateral=True),aelt,adeg))
aexpr.vector().set_local(aval)
A = Function(V, name="coef")
A.assign(assemble(interpolate(aexpr, V)))

#plt.clf()
#fig, axes = plt.subplots()
#collection = tripcolor(A, axes=axes)
#fig.colorbar(collection);
#plt.title('coefficient')
#plt.savefig(coefplotfile, dpi=300)
#PETSc.Sys.Print("> coefficient plotted to {}".format(coefplotfile))

with CheckpointFile(coefh5file,'w') as afile:
    afile.save_mesh(mesh)
    afile.save_function(A)
