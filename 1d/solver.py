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

def makedir(resultsdir='Results'):
    """
    Check that the directory resultsdir exists, and, if so, create a new subdirectory
    for the output using the first available name from 000000, 000001, ...  Then return its name.
    """
    assert os.path.isdir(resultsdir), 'The subdirectory {} must exist in the current directory'.format(resultsdir)
    id = 0
    while os.path.exists(resultsdir + '/{:06d}'.format(id)):
        id += 1
    outdir = resultsdir + '/{:06d}'.format(id)
    os.mkdir(outdir)
    print('> subdirectory name:{:06d}'.format(id), flush=True)
    return outdir
    
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
def combine_images(columns, space, images,file):
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    width_max = max([Image.open(image).width for image in images])
    height_max = max([Image.open(image).height for image in images])
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = Image.new('RGBA', (background_width, background_height), (255, 255, 255, 255))
    x = 0
    y = 0
    for i, image in enumerate(images):
        img = Image.open(image)
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
    background.save(file)

def eigen_solver(mesh,A,deg,nreq,target,bctype):
    # Find the first nreq eigenpaires nearest the given target
    V = FunctionSpace(mesh, "Lagrange", deg)
    u = TrialFunction(V)
    v = TestFunction(V)
    b = A*dot(grad(u), grad(v))*dx
    m = u*v*dx
    uh = Function(V)
    if bctype == 'dirichlet':
       boundary_ids = (1,2) # 1: left endpoint, 2: right endpoint
       bc = DirichletBC(V, 0,boundary_ids)
       B = assemble(b, bcs=bc)
       M = assemble(m, bcs=bc, weight=0.)
    else:    
       B = assemble(b)
       M = assemble(m)
    Bsc, Msc = B.M.handle, M.M.handle
    # create SLEPc eigensolver
    Eps = SLEPc.EPS().create()
    Eps.setOperators(Bsc, Msc)
    # Set problem type to be generalized Hermitian
    Eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    Eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    Eps.setDimensions(nreq)
    # closest to target (in magnitude).
    Eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    # target -- determine the portion of the spectrum of interest
    Eps.setTarget(target)
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
    Eps.solve()
    nconv = Eps.getConverged()
    print(f"> computed {nconv} eigenvalues.")
    return Eps, nconv, Bsc,V
    
def get_eigenpairs(Eps,nconv,Bsc,V,x0,x1,nelts,npts,plotefuns,eigenvalfile,eigenfunplotfile,eigenfunmontagefile):
    # get eigenpairs
    eigenvalues = []
    eigenf_imgs = []
    for i in range(nconv):
        r = Eps.getEigenvalue(i).real
        #print("{:12.9f}".format(r))
        eigenvalues.append(r)
        # get eigenfunction
        rxv, _ = Bsc.getVecs()
        cxv, _ = Bsc.getVecs()
        r = Eps.getEigenpair(i, rxv, cxv)
        rx = rxv.array
        # normalize eigenfunction so max = max magnitude = 1
        rxmax = rx.max()
        rxmin = rx.min()
        if rxmax < -rxmin:
            rx = rx/rxmin
        else:
            rx = rx/rxmax
        eigenfun = Function(V)
        eigenfun.vector().set_local(rx)
        if i in plotefuns:
            x = np.linspace(x0, x1, npts, endpoint=False)
            y = eval_u(eigenfun,x)
            plt.clf()
            plt.plot(x, y, alpha=.75, linewidth=2)
            plt.xlim([x0, x1])
            plt.ylim([-1.1, 1.1])
            plt.title('nelts={}  eigenfunction {}  $\lambda=${:7.5f}'.format(nelts, i, r.real))
            print("> eigenfunction {} plotted to ".format(i) + eigenfunplotfile.format(i))
            plt.savefig(eigenfunplotfile.format(i), dpi=500)
            eigenf_imgs.append(eigenfunplotfile.format(i))
  
    np.savetxt(eigenvalfile, eigenvalues)
    print("> eigenvalues written to {}".format(eigenvalfile))
    combine_images(columns=5, space=20, images=eigenf_imgs,file=eigenfunmontagefile)
    print("> eigenfunction montage written to {}".format(eigenfunmontagefile)) 


def plot_coeff(x0,x1,A,npts,filename):
    # evaluate coefficient, save to file and plot
    plt.clf()
    print("> evaluating coefficient")
    pts = np.linspace(x0, x1, npts, endpoint=True)
    avals = eval_u(A,pts)
    plt.plot(pts, avals, alpha=.75, linewidth=2)
    plt.xlim([x0, x1])
    plt.title('coefficient')
    plt.savefig(filename, dpi=500)
    print("> coefficient plotted to {}".format(filename))
