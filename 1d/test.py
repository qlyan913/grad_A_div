"""
   Plot the coefficient function:
   A(x) = 1/(1+ sum_{integer n: x0<= n <= x1}f(x-n-dn(w))
   f = 1/8[max{(1-x^2/s^2)^3,0}(3x^2+1)]', supp(f) in [-s,s]
   dn uniform distribution on [-dmax,dmax]
   We choose s=1/4 and dmax=1/5 such that s+dmax<1/2
"""
import os,math, json
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from slepc4py import SLEPc
import numpy as np
from solver import *
nelts=400
npts=1000
x0=-0.6
x1=0.6
# plot of  f
fplotfile="test_figs/f.png"
mesh = IntervalMesh(nelts, x0, x1)
x , = SpatialCoordinate(mesh)
dmax=0.2
s=0.25
#fexpr= conditional(abs(x)>s,0,6/8*x*(1-pow(x,2)/pow(s,2))**3-6/(8*pow(s,2))*x*(1-pow(x,2)/pow(s,2))**2*(3*x**2+1))
fexpr=conditional(abs(x)>s,0,(1-pow(x,2)/pow(s,2))**3*(3*pow(x,2)+1))
aelt='CG'
adeg=3
F = assemble(interpolate(fexpr, FunctionSpace(mesh, aelt, adeg)))
plt.clf()
print("> evaluating f")
pts = np.linspace(x0, x1, npts, endpoint=True)
avals = eval_u(F,pts)
plt.plot(pts, avals, alpha=.75, linewidth=2)
plt.xlim([x0, x1])
plt.savefig(fplotfile, dpi=500)
print("> f plotted to {}".format(fplotfile))

# plot coefficients
x0=0
x1=500
fsumplotfile="test_figs/fsum.png"
coefplotfile="test_figs/coeff.png"
mesh = IntervalMesh(nelts, x0, x1)
x , = SpatialCoordinate(mesh)
nn=x1-x0-1
dn=-dmax+np.random.rand(nn)*(2*dmax)
f_sum=0.0
for i in range(nn):
   x_center=i+1+dn[i]
   #f_sum=f_sum + conditional(abs(x-x_center)>s,0,6/8*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**3-6/(8*pow(s,2))*(x-x_center)*(1-pow(x-x_center,2)/pow(s,2))**2*(3*pow(x-x_center,2)+1))
   f_sum=f_sum+ conditional(abs(x-x_center)>s,0,(1-pow(x-x_center,2)/pow(s,2))**3*(3*pow(x-x_center,2)+1))
F = assemble(interpolate(f_sum, FunctionSpace(mesh, aelt, adeg)))
aelt='CG'
adeg=3
plt.clf()
print("> evaluating f sum")
pts = np.linspace(x0, x1, npts, endpoint=True)
avals = eval_u(F,pts)
plt.plot(pts, avals, alpha=.75, linewidth=2)
plt.xlim([x0, x1])
plt.savefig(fsumplotfile, dpi=500)
print("> f sum plotted to {}".format(fsumplotfile))

aexpr=1./(1+f_sum)
A = assemble(interpolate(aexpr, FunctionSpace(mesh, aelt, adeg)))
plt.clf()
print("> evaluating coeff")
pts = np.linspace(x0, x1, npts, endpoint=True)
avals = eval_u(A,pts)
plt.plot(pts, avals, alpha=.75, linewidth=2)
plt.xlim([x0, x1])
plt.savefig(coefplotfile, dpi=500)
print("> f sum plotted to {}".format(coefplotfile))

