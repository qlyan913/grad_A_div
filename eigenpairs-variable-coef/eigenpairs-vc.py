#!/usr/bin/env python3
"""
Douglas N. Arnold, 2021-07-07

For a specified coefficient a, interval, boundary conditions, etc., the
program eigenpairs-vc.py computes the desired number of eigenpairs nearest
a given target for the variable coefficient problem

  - (a u')' = lambda u.

"""
import sys, slepc4py, os
slepc4py.init(sys.argv)
from dolfin import *
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters
import sys, os, json, argparse
from petsc4py import PETSc
from slepc4py import SLEPc
from socket import gethostname
from itertools import groupby

starttime0 = datetime.now()
rank = PETSc.COMM_WORLD.Get_rank()
host = gethostname()
nprocs = PETSc.COMM_WORLD.Get_size()

def parse_hyphen_ranges(s):
    # takes a string giving a comma separated list of numbers and hyphenated ranges and
    # converts to a list of numbers.  E.g.,
    # "5" -> [5] ; "0,2-4,6-8" -> [0, 2, 3, 4, 6, 7, 8]
    if s == '':
        return []
    else:
        s="".join(s.split())#removes white space
        r=set()
        for x in s.split(','):
            t=x.split('-')
            if len(t) not in [1,2]: raise SyntaxError("hash_range is given its arguement as "+s+" which seems not correctly formated.")
            r.add(int(t[0])) if len(t)==1 else r.update(set(range(int(t[0]),int(t[1])+1)))
        l=list(r)
        l.sort()
        return l

def eval_u(u, pts):
    """
    evaluate finite element function at an array of points

    input values:
        u: finite element function
        pts: array of points

    return values:
        vals: array of values of u
    """
    return np.array([u(x) for x in pts])

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
    print('{:06d}'.format(id), flush=True)
    return outdir

def strtonum(string):
    # convert string to integer or float, depending on whether the string contains a decimal point
    if '.' in string:
        return float(string)
    else:
        return int(string)

def logevent(msg):
    # write message to logfile
    logf.write(msg + '\n')

def printandlog(msg):
    # write message to logfile and print to screen
    logf.write(msg + '\n')
    print(msg)

# Given a elements (tuple, list, array) return the length of the longest
# subsequence of consecutive zeros (an element is considered a zero if "x == 0" returns True)
def max_run_length(seq):
    if 0 in seq:
        return max(sum(1 for _ in run) for val, run in groupby(seq) if val == 0)
    else:
        return 0

# parse args and set parameter, write them to a file
parser = argparse.ArgumentParser()
parser.add_argument('--pdir', type=str, default='Results', help='parent dir')
parser.add_argument('--nelts', type=int, default=32, help='no. of elements')
parser.add_argument('--coef', type=str, default='constant', help='coefficient type')
parser.add_argument('--bc', type=str, default='periodic', help='boundary conditions')
# note: to handle negative x0, use something like "--x0 ' -1.'" (note the space)
parser.add_argument('--x0', type=float, default=0.0, help='left endpoint')
parser.add_argument('--x1', type=float, default=0.0, help='right endpoint')
parser.add_argument('--deg', type=int, default=1, help='element degree')
parser.add_argument('--npts', type=int, default=1025, help='evaluation points')
parser.add_argument('--params', nargs='*', type=strtonum, default=[], help='parameters in coefficient')
parser.add_argument('--nreq', type=int, default=10, help='no. of eigenvalues')
parser.add_argument('--target', type=float, default=0., help='target')
parser.add_argument('--saveefuns', type=str, default='', help='eigenfunctions to save')
parser.add_argument('--plotefuns', type=str, default='', help='eigenfunctions to plot')
parser.add_argument('--verbose', help='print more data', action='store_true')
args = parser.parse_args()
pdir = args.pdir
nelts = args.nelts
coeftype = args.coef
bctype = args.bc
x0 = args.x0
x1 = args.x1
deg = args.deg
npts = args.npts
params = args.params
nreq = args.nreq
target = args.target
# saveefuns argument is a comma separated list of numbers and ranges, e.g.,
# 5 or 1-10 or 0,2,5-9,15-20.  Parse it to a list of integers.
saveefunsranges = args.saveefuns
saveefuns = parse_hyphen_ranges(saveefunsranges)
plotefunsranges = args.plotefuns
plotefuns = parse_hyphen_ranges(plotefunsranges)
verbose = args.verbose

# canonicalize coeftype and bctype strings
coeftype = coeftype.lower()
bctype = bctype.lower()
if coeftype not in ('constant', 'threeconstant', 'uniform', 'binomial'):
    print('Coefficient type {} unknown.  Using constant 1 as coefficient'.format(coeftype))
    coeftype = 'constant'
    params = [1.]
if bctype not in ('dirichlet', 'neumann', 'periodic'):
    print('Boundary condition type {} unknown.  Using periodic.'.format(bctype))
    bctype = 'periodic'
# store parameters in dictionary
runparameters = {
    'bctype': bctype,
    'deg': deg,
    'nelts': nelts,
    'npts': npts,
    'params': params,
    'coeftype': coeftype,
    'x0': x0,
    'x1': x1,
    }

# resolution for plot files
res = 600

# create directory and filenames for output
outdir = makedir(resultsdir=pdir)
# filenames
logfile = outdir + '/' + 'Log.txt'
paramfile = outdir + '/' + 'Parameters.json'
coeffile = outdir + '/' + 'coefficient.txt.gz'
coefvalfile = outdir + '/' + 'coefficientvals.txt.gz'
coefplotfile = outdir + '/' + 'coefficient.png'
eigenvalfile = outdir + '/' + 'eigenvalues.txt'
eigenfunfile = outdir + '/' + 'eigenfun{:05d}.txt.gz'
eigenfunplotfile = outdir + '/' + 'eigenfun{:05d}.png'
    
# open log file and record program name, time, working directory, and command line
logf = open(logfile, 'w')
logevent('***' + os.path.basename(__file__))
logevent('run start time: ' + starttime0.strftime('%c'))
logevent('working directory: ' + os.getcwd())
logevent('command line: ' + ' '.join(sys.argv) + '\n')
printandlog('> directory created for results: ' + outdir)

# write parameters to file
paramf = open(paramfile, 'w')
json.dump(runparameters, paramf, indent=4)
paramf.write('\n')
paramf.close()
printandlog("> run parameters written to {}".format(paramfile))

# create the finite element mesh
mesh = IntervalMesh(nelts, x0, x1)

# define coefficient as a finite element function
if coeftype == 'constant':
    # constant coefficient, 1 parameter
    #   params[0]: value (default = 0.)
    if len(params) > 0:
        aval = params[0]
    else:
        aval = 0.
    aexpr = Constant(aval)
    aelt = 'DG'
    adeg = 0
    a = interpolate(aexpr, FunctionSpace(mesh, aelt, adeg))
elif coeftype == 'threeconstant':
    # coefficient is pw constant with 3 pieces, includes square well and square barrier coefficient
    # V = b0, x0 < x < a0;  b1, a0 < x < a1;  b2, a1 < x < x1
    assert len(params) == 5, "For threeconst coefficient, 5 parameters required"
    # read the parameters and create the mesh
    a0 = params[0]
    a1 = params[1]
    b0 = params[2]
    b1 = params[3]
    b2 = params[4]
    # initialize 3 element mesh
    mesh0 = Mesh()
    med = MeshEditor()
    med.open(mesh0, "interval", 1, 1)
    # add vertices
    med.init_vertices(4)
    for i, v in enumerate([x0, a0, a1, x1]):
        med.add_vertex(i, [v])
    # add cells
    med.init_cells(3)
    for i in range(3):
        c = (i, i + 1)
        med.add_cell(i, np.array([i, i + 1]))
    med.close()
    assert mesh0.ordered()
    # make pw constant with 3 pieces
    aelt = 'DG'
    apdeg = 0
    p0 = Function(FunctionSpace(mesh0, aelt, adeg))
    p0.vector().set_local([b0, b1, b2])
    # interpolate to pw constant with nelts pieces
    a = interpolate(a0, FunctionSpace(mesh, aelt, adeg))
elif coeftype == 'uniform':
    # pw constant with iid random values from uniform distribution
    # nc pieces, values in [amin, amax]
    # 4 parameters: nc, amin, amax, seed (default = 32, 0.5, 2., None)
    if len(params) > 0:
        nc = int(params[0])
    else:
        nc = 32
    assert nelts % nc == 0, 'nelts must be a multiple of nc'
    if len(params) > 1:
        amin = float(params[1])
    else:
        amin = 0.5
    if len(params) > 2:
        amax = float(params[2])
    else:
        amax = 2.0
    if len(params) > 3:
        seed = int(params[3])
    else:
        seed = None
    np.random.seed(seed)
    aval = amin + np.random.rand(nc) * (amax - amin)
    aelt = 'DG'
    adeg = 0
    # create pw constant with nc pieces
    a0 = Function(FunctionSpace(IntervalMesh(nc, x0, x1), aelt, adeg))
    a0.vector().set_local(aval)
    # interpolate to pw constant with nelts pieces
    a = interpolate(a0, FunctionSpace(mesh, aelt, adeg))
elif coeftype == 'binomial':
    # pw constant with iid random values from binomial distribution
    # nc pieces, values in {amin, amax} with probabilities 1-prob, prob
    # 4 parameters: nc, amin, amax, prob, seed (default = 32, .5, 2.0, 0.5, None)
    if len(params) > 0:
        nc = int(params[0])
    else:
        nc = 32
    assert nelts % nc == 0, 'nelts must be a multiple of nc'
    if len(params) > 1:
        amin = float(params[1])
    else:
        amin = 0.5
    if len(params) > 2:
        amax = float(params[2])
    else:
        amax = 2.0
    if len(params) > 3:
        prob = params[3]
    else:
        prob = 0.5
    if len(params) > 4:
        seed = int(params[4])
    else:
        seed = None
    np.random.seed(seed)
    aval = amin + np.random.binomial(1, prob, nc) * (amax - amin)
    np.savetxt(coefvalfile, aval)
    printandlog("> coefficient values written to {}".format(coefvalfile))
    aelt = 'DG'
    adeg = 0
    # create pw constant with nc pieces
    a0 = Function(FunctionSpace(IntervalMesh(nc, x0, x1), aelt, adeg))
    a0.vector().set_local(aval)
    # interpolate to pw constant with nelts pieces
    a = interpolate(a0, FunctionSpace(mesh, aelt, adeg))

# evaluate coefficient, save to file and plot
plt.clf()
printandlog("> evaluating coefficient")
if bctype == 'periodic':
    pts = np.linspace(x0, x1, npts, endpoint=False)
else:
    pts = np.linspace(x0, x1, npts, endpoint=True)
avals = eval_u(a, pts)
np.savetxt(coeffile, avals)
printandlog("> coefficient written to {}".format(coeffile))
plt.plot(pts, avals, alpha=.75, linewidth=2)
plt.xlim([x0, x1])
plt.title('coefficient'.format(nelts, deg))
plt.savefig(coefplotfile, dpi=res)
printandlog("> coefficient plotted to {}".format(coefplotfile))
# create the finite element space
if bctype in ('dirichlet', 'neumann'):
    V = FunctionSpace(mesh, 'Lagrange', deg)
elif bctype == 'periodic':
    class PeriodicBoundary(SubDomain):
        # Target domain which DOF can be specified is the left-end point
        def inside(self, x, on_boundary):
            return (near(x[0], x0) and on_boundary)
        # Map right end point to left end point
        def map(self, x, y):
            y[0] = x[0] - x1 + x0
    pbc = PeriodicBoundary()
    V = FunctionSpace(mesh, 'Lagrange', deg, constrained_domain=pbc)
# assemble the stiffness and mass matrices
u = TrialFunction(V)
v = TestFunction(V)
b = (a * inner(grad(u), grad(v)))*dx(domain=V.mesh())
m = u*v*dx
if bctype == 'dirichlet':
    dummy = Constant(0.)*v*dx(domain=V.mesh())
    bc = DirichletBC(V, 0., DomainBoundary())
    A, _ = assemble_system(b, dummy, bc)
    M, _ = assemble_system(m, dummy, bc)
    # set the diagonal elements of M corresponding to boundary nodes to zero to
    # remove spurious eigenvalues.
    bc.zero(M)
    A = as_backend_type(A).mat() # convert from dolfin.cpp.la.Matrix to petsc4py.PETSc.Mat
    M = as_backend_type(M).mat() # convert from dolfin.cpp.la.Matrix to petsc4py.PETSc.Mat
else:
    A = PETScMatrix(MPI.comm_self)
    assemble(b, tensor=A)
    A = A.mat() # convert from dolfin.cpp.la.PETScMatrix to petsc4py.PETSc.Mat
    M = PETScMatrix(MPI.comm_self)
    assemble(m, tensor=M)
    M = M.mat() # convert from dolfin.cpp.la.PETScMatrix petsc4py.PETSc.Mat
printandlog("> stiffness matrix of size {} x {} created".format(A.size[0], A.size[1]))
printandlog("> mass matrix of size {} x {} created".format(M.size[0], M.size[1]))
matrixtime = datetime.now()
printandlog("+ coefficient and matrix construction: {} seconds".format((matrixtime - starttime0).seconds))

# Set up a (parallel) SLEPc eigensolver to find the first nreq eigenvalues
# of the generalized Hermitian eigenvalue problem.
# It is necessary to set the solver type to
# Krylov-Schur and use a shift-and-invert spectral transform with
# a direct Cholesky factorization as the solver.  I use PETSc's Cholesky
# solver, although I could use MUMPS. ::

# create eigensolver
starttime = datetime.now()
eps = SLEPc.EPS()
eps.create(comm=PETSc.COMM_WORLD)
eps.setOperators(A, M)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
eps.setTarget(target)
eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
eps.setDimensions(nreq)
# set the spectral transform of the EPS to shift-and-invert
st = eps.getST()
st.setType(SLEPc.ST.Type.SINVERT)
# set the solver for the spectral transform to direct: Cholesky
ksp = st.getKSP()
ksp.setType(PETSc.KSP.Type.PREONLY)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.CHOLESKY)

# Finish setting up the eigenvalue solver::

# set parameters from command line
#eps.setFromOptions()
#eps.setUp()

# Solve the eigenvalue problem::
eps.solve()
nconv = eps.getConverged()
if rank == 0:
    solvetime = datetime.now()
    printandlog("> computed {} eigenvalues".format(nconv))
    print("+ eigenvalue solve: {} seconds".format((solvetime - starttime).seconds))
#eps.view()

# Print some output.  Each process prints the eigenvalues
# it computed.  Rest is printed only by process 0. ::

if rank == 0:
    if verbose:
        header = """\
** Run data **
host: {}
starttime: {}
problem size: {}
number of MPI processes: {}
number of iterations (proc 0): {}
problem type: {}
solution method: {}
spectral transform: {}
requested eigenvalues: {}
converged eigenvalues: {}
total time: {:.1f}""".format(host, starttime.ctime(), A.size, nprocs, eps.getIterationNumber(), \
          eps.getProblemType(), eps.getType(), \
          eps.getST().getType(), nreq, nconv, (solvetime - starttime).seconds)
        printandlog(header)
        printandlog("** Parameters **")
        printandlog(str(params))
    eigenvalues = []
    for i in range(nconv):
        r = eps.getEigenvalue(i).real
        #print("{:12.9f}".format(r))
        eigenvalues.append(r)
        # get eigenfunction
        rxv, _ = A.getVecs()
        cxv, _ = A.getVecs()
        r = eps.getEigenpair(i, rxv, cxv)
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
        # compute the participation ratio
        eigenfun2 = eigenfun*eigenfun
        eigenfun4 = eigenfun2*eigenfun2
        if i in saveefuns or i in plotefuns:
            x = np.linspace(x0, x1, npts, endpoint=False)
            y = eval_u(eigenfun, x)
        if i in saveefuns:
            np.savetxt(eigenfunfile.format(i), y)
            printandlog("> eigenfunction {} saved to ".format(i) + eigenfunfile.format(i))
        if i in plotefuns:
            plt.clf()
            plt.plot(x, y, alpha=.75, linewidth=2)
            plt.xlim([x0, x1])
            plt.ylim([-1.1, 1.1])
            plt.title('nelts={}  eigenfunction {}  $\lambda=${:7.5f}'.format(nelts, i, r.real))
            printandlog("> eigenfunction {} plotted to ".format(i) + eigenfunplotfile.format(i))
            plt.savefig(eigenfunplotfile.format(i), dpi=res)
    # print and save eigenvalues to file
    print(eigenvalues)
    np.savetxt(eigenvalfile, eigenvalues)
    print("> eigenvalues written to {}".format(eigenvalfile))
    specsavetime = datetime.now()
    printandlog("+ saving spectral info: {} seconds".format((specsavetime - solvetime).seconds))

endtime = datetime.now()
printandlog("+ total time: {} seconds".format((endtime - starttime0).seconds))


logf.write('\n')
logf.close()
