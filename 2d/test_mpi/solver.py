"""
Codes copied from eigenpairs-vc.py by Douglas Arnold, 2021-07-07

Solve the eigenvalue problem with variable coefficient:
   -(Au')'=lambda u
on an interval.

Computes nre eigenpairs nearest a given target.

"""
import os
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from slepc4py import SLEPc
import numpy as np
from datetime import datetime

def eval_u(u, pts):
    """
    evaluate finite element function at an array of points

    input values:
        u: finite element function
        pts: array of points

    return values:
        vals: array of values of u
    """
    return np.array([u.at(x) for x in pts])

from PIL import Image

def eigen_solver_slicing(mesh,A,deg,sigma_0,sigma_1,bctype,flag=1):
    # find all the eigenvalues of the generalized Hermitian eigenvalue problem which belong to the interval [sigma_0, sigma_1]
    """
     flag ---- 1: -div A grad phi = lambda phi
          ---- 2: -div A grad phi = lambda A phi
          ---- 3: -div grad phi = lambda A phi
    """
    # Find the first nreq eigenpaires nearest the given target
    rank = PETSc.COMM_WORLD.Get_rank()
    size = PETSc.COMM_WORLD.Get_size()
    subints = np.linspace(sigma_0, sigma_1, size + 1)
#    if rank == 0:
#       starttime = datetime.now()
    V = FunctionSpace(mesh, 'Lagrange', deg)
    PETSc.Sys.Print("> degree of freedom: ", V.dof_dset.layout_vec.getSize())
    u = TrialFunction(V)
    v = TestFunction(V)
    if flag == 1:
       b = A*dot(grad(u), grad(v))*dx
       m = u*v*dx
    elif flag == 2:
       b = A*dot(grad(u), grad(v))*dx
       m = A*u*v*dx
    elif flag == 3:
       b = dot(grad(u), grad(v))*dx
       m = A*u*v*dx
    uh = Function(V)
    if bctype == 'dirichlet':
       boundary_ids = (1,2,3,4) #
       bc = DirichletBC(V, 0,boundary_ids)
       B = assemble(b, bcs=bc)
       M = assemble(m, bcs=bc, weight=0.)
    else:    
       B = assemble(b)
       M = assemble(m)
    Bsc, Msc = B.M.handle, M.M.handle
    B_petsc = Bsc.convert('mpisbaij')
    M_petsc = Msc.convert('mpisbaij')
  # create SLEPc eigensolver
    Eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    Eps.setOperators(B_petsc, M_petsc)
    # Set problem type to be generalized Hermitian
    Eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    Eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    Eps.setInterval(sigma_0,sigma_1)
    Eps.setWhichEigenpairs(SLEPc.EPS.Which.ALL)
    Eps.setKrylovSchurPartitions(size)
    Eps.setKrylovSchurSubintervals(subints)
    # set the spectral transform of the EPS to shift-and-invert
    ST = Eps.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)
    ksp = ST.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.CHOLESKY)
   # PC = ST.getKSP().getPC()
   # PC.setType("lu")
   # PC.setFactorSolverType("mumps")
    Eps.setST(ST)
    PETSc.Sys.Print("soling the eigen value")
    Eps.solve()
 #   if rank == 0:
 #      endtime = datetime.now()
 #      elapsed = endtime - starttime
 #      totalsecs = elapsed.seconds
#   PETSc.Sys.Print('used {} seconds to solve'.format(totalsecs))
    nconv = Eps.getConverged()
    PETSc.Sys.Print(f"> computed {nconv} eigenvalues.")
    q = Eps.getKrylovSchurSubcommInfo()
    print("Proc {}:  {} of {} eigenvalues\n".format(q[0], q[1], nconv))
    return Eps, nconv, Bsc,V


def plot_coeff(x0,x1,A,npts,filename):
    # evaluate coefficient, save to file and plot
    plt.clf()
    print("> evaluating coefficient")
    pts = np.linspace(x0, x1, npts, endpoint=True)
    avals = eval_u(A,pts)
    plt.plot(pts, avals, alpha=.75, linewidth=2)
    plt.xlim([x0, x1])
    plt.title('coefficient')
    plt.savefig(filename, dpi=300)
    print("> coefficient plotted to {}".format(filename))
