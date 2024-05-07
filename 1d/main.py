"""
Solve the eigenvalue problem with variable coefficient:
   -(Au')'=lambda u
"""
import matplotlib.pyplot as plt
from firedrake import *
from firedrake.petsc import PETSc
from firedrake.__future__ import interpolate
from slepc4py import SLEPc
import numpy as np
deg = 8 
nelts=1024
x0=0
x1=1024
def eigen_solver(mesh,A,deg):
    V = FunctionSpace(mesh, "Lagrange", deg)
    u = TrialFunction(V)
    v = TestFunction(V)
    b = A*dot(grad(u), grad(v))*dx
    m=u*v*dx
    uh = Function(V)
    boundary_ids = (1,2) # 1: left endpoint, 2: right endpoint
    bc = DirichletBC(V, 0,boundary_ids)
    B = assemble(b, bcs=bc)
    M = assemble(m, bcs=bc, weight=0.)
    Bsc, Msc = B.M.handle, M.M.handle
    E = SLEPc.EPS().create()
    E.setType(SLEPc.EPS.Type.ARNOLDI)
    E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    E.setDimensions(1, SLEPc.DECIDE)
    E.setOperators(Bsc, Msc)
    ST = E.getST()
    ST.setType(SLEPc.ST.Type.SINVERT)
    PC = ST.getKSP().getPC()
    PC.setType("lu")
    PC.setFactorSolverType("mumps")
    E.setST(ST)
    E.solve()
    vr, vi = Bsc.getVecs()
    with uh.dat.vec_wo as vr:
        lam = E.getEigenpair(0, vr, vi)
    return lam, uh

mesh = IntervalMesh(nelts, x0, x1)
aval=1
aexpr = Constant(aval)
aelt = 'DG'
adeg = 0
A = assemble(interpolate(aexpr, FunctionSpace(mesh, aelt, adeg)))
lam, uh = eigen_solver(mesh,A,deg)
print(lam)
