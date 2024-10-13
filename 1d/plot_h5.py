import matplotlib.pyplot as plt
from firedrake import *
import numpy as np
from solver import *
L=200
x0=0
x1=L
npts=2*L
nn=5 # number of eigenfunctions
# create directory and filenames for output
outdir = 'Results/000066'
eigenfunplotfile="eigen.png"
eigenvalfile = outdir + '/' + 'eigenvalues.txt'
eigenfunh5file  = outdir + '/h5_file/' + 'eigenfun{:05d}.h5'
eigen_list=np.loadtxt(eigenvalfile)
with CheckpointFile(eigenfunh5file.format(nn),'r') as afile:
     mesh=afile.load_mesh()
     eigenfun=afile.load_function(mesh,"eigen")

x = np.linspace(x0, x1, npts, endpoint=False)
y = eval_u(eigenfun,x)
f2=assemble(eigenfun**2*dx)
f4=assemble(eigenfun**4*dx)
pr=1/(x1-x0)*(f2**2)/f4
plt.clf()
plt.plot(x,y, alpha=.75, linewidth=2)
plt.xlim([x0, x1])
plt.ylim([-1.1, 1.1])
plt.title('eigenfunction {}  $\lambda=${:7.5f} ratio {:1.5f}'.format(nn,eigen_list[nn],pr))
print("> eigenfunction {} plotted to ".format(nn)+eigenfunplotfile)
plt.savefig(eigenfunplotfile, dpi=300)


