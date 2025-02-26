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
    os.mkdir(outdir+'/h5_file/')
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

def eigen_solver(mesh,A,deg,nreq,target,bctype,flag=1):
    """
     flag ---- 1: -div A grad phi = lambda phi
          ---- 2: -div A grad phi = lambda A phi
          ---- 3: -div grad phi = lambda A phi
    """
    # Find the first nreq eigenpaires nearest the given target
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
   # pc = ksp.getPC()
   # pc.setType(PETSc.PC.Type.CHOLESKY)
    pc = ST.getKSP().getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    Eps.setST(ST)
    Eps.solve()
    nconv = Eps.getConverged()
    PETSc.Sys.Print(f"> computed {nconv} eigenvalues.")
    return Eps, nconv, Bsc,V

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
    if rank == 0:
       starttime = datetime.now()
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
    PETSc.Sys.Print("> solving the eigen value")
    Eps.solve()
    if rank == 0:
       endtime = datetime.now()
       elapsed = endtime - starttime
       totalsecs = elapsed.seconds
    PETSc.Sys.Print('used {} seconds to solve'.format(totalsecs))
    nconv = Eps.getConverged()
    PETSc.Sys.Print(f"> computed {nconv} eigenvalues.")
    q = Eps.getKrylovSchurSubcommInfo()
    print("Proc {}:  {} of {} eigenvalues\n".format(q[0], q[1], nconv))
    return Eps, nconv, Bsc,V

def get_eigenpairs(Eps,nconv,Bsc,V,L,plotefuns,plotefuns_2,eigenvalfile,eigenfunplotfile,eigenfunmontagefile,eigenfunmontagefile_2,eigenfun_smpr_file,center_list=[],flag=0,eigenfunmon_all="",n_all=500):
    # get eigenpairs
    eigenvalues = []
    eigenvalues_v2 = []
    pratio=[]
    modes=[]
    eigenf_imgs = []
    eigenf_imgs_2 = []
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
        eigenfun = Function(V,name="eigen")
        eigenfun.vector().set_local(rx)
        if flag == 0:
           if i in plotefuns or i in plotefuns_2:
              eigenvalues_v2.append(r.real)
              modes.append(i)
              f2=assemble(eigenfun**2*dx)
              f4=assemble(eigenfun**4*dx)
              pr=1/(L**2)*(f2**2)/f4
              pratio.append(pr)
              plt.clf()
              fig, axes = plt.subplots()
              collection = tripcolor(eigenfun, axes=axes)
              fig.colorbar(collection);
              plt.title(' eigenfunction {}  $\lambda=${:7.5f}  ratio {:1.5f}'.format( i, r.real,pr))
              PETSc.Sys.Print("> eigenfunction {} plotted to ".format(i) + eigenfunplotfile.format(i))
              plt.savefig(eigenfunplotfile.format(i), dpi=300)
              plt.close()
              if i in plotefuns:
                 eigenf_imgs.append(eigenfunplotfile.format(i))
              if i in plotefuns_2:
                 eigenf_imgs_2.append(eigenfunplotfile.format(i))
              if pr <0.07:
                   plt.clf()
                   fig = plt.figure()
                   axes= fig.add_subplot(projection="3d")
                   collection=trisurf(eigenfun,axes=axes);
                   fig.colorbar(collection);
                   plt.title('eigenfunction {}  $\lambda=${:7.5f} ratio {:1.5f}'.format( i, r.real,pr))
                   PETSc.Sys.Print("> eigenfunction {} with p-ration{:1.5f} plotted to ".format(i,pr) + eigenfun_smpr_file.format(i))
                   plt.savefig(eigenfun_smpr_file.format(i),dpi=300)
                   plt.close()
        else:
            if i < n_all+1:
               eigenvalues_v2.append(r.real)
               modes.append(i)
               f2=assemble(eigenfun**2*dx)
               f4=assemble(eigenfun**4*dx)
               pr=1/(L**2)*(f2**2)/f4
               pratio.append(pr)
               plt.clf()
               fig, axes = plt.subplots()
               collection = tripcolor(eigenfun, axes=axes)
               fig.colorbar(collection);
               plt.title('eigenfunction {}  $\lambda=${:7.5f} ratio {:1.5f}'.format( i, r.real,pr))
               PETSc.Sys.Print("> eigenfunction {} plotted to ".format(i) + eigenfunplotfile.format(i))
               plt.savefig(eigenfunplotfile.format(i), dpi=300)
               plt.close()
               if pr <0.07:
                   plt.clf()
                   fig = plt.figure()
                   axes= fig.add_subplot(projection="3d")
                   collection=trisurf(eigenfun,axes=axes);
                   fig.colorbar(collection);
                   plt.title('eigenfunction {}  $\lambda=${:7.5f} ratio {:1.5f}'.format( i, r.real,pr))
                   PETSc.Sys.Print("> eigenfunction {} with p-ration{:1.5f} plotted to ".format(i,pr) + eigenfun_smpr_file.format(i))
                   plt.savefig(eigenfun_smpr_file.format(i),dpi=300)
                   plt.close()
            else:
               break
    np.savetxt(eigenvalfile, eigenvalues)
    print("> eigenvalues written to {}".format(eigenvalfile))
    if flag == 0:
       combine_images(columns=5, space=20, images=eigenf_imgs,file=eigenfunmontagefile)
       PETSc.Sys.Print("> eigenfunction montage written to {}".format(eigenfunmontagefile)) 
       combine_images(columns=5, space=20, images=eigenf_imgs_2,file=eigenfunmontagefile_2)
       PETSc.Sys.Print("> another eigenfunction montage written to {}".format(eigenfunmontagefile_2)) 
    else:
       for i in range(0,n_all,25):
           segment=list(range(i,i+25))
           i0=segment[0]
           iend=segment[-1]
           eigenf_imgs=[]
           for j in segment:
              eigenf_imgs.append(eigenfunplotfile.format(j))
           combine_images(columns=5, space=20, images=eigenf_imgs,file=eigenfunmon_all.format(i0,iend))
           PETSc.Sys.Print("> eigenfunction montage between {} and {} is  written to {}".format(i0,iend,eigenfunmon_all.format(i0,iend)))
    return modes, eigenvalues_v2, pratio

def  get_eigenpairs_v2(Eps,nreq,Bsc,V,L,plotefuns,eigenfunplotfile,eigenfunmontagefile,eigenfun_smpr_file,target,mesh=[],eigenfunh5file=[]):
   # get eigenpairs
    eigenvalues = []
    eigenvalues_v2 = []
    pratio=[]
    modes=[]
    eigenf_imgs = []
    eigenf_imgs_smpr = []
    targets=[]
    for i in range(nreq):
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
           eigenvalues_v2.append(r.real)
           modes.append(i)
           targets.append(target)
           f2=assemble(eigenfun**2*dx)
           f4=assemble(eigenfun**4*dx)
           pr=1/(L**2)*(f2**2)/f4
           pratio.append(pr)
           plt.clf()
           fig, axes = plt.subplots()
           collection = tripcolor(eigenfun, axes=axes,cmap='bwr')
           collection.set_clim(-1.1,1.1)
           fig.colorbar(collection);
           plt.title(' eigenfunction {}  $\lambda=${:7.5f}  ratio {:1.5f}'.format( i, r.real,pr))
           PETSc.Sys.Print("> eigenfunction {} plotted to ".format(i) + eigenfunplotfile.format(target,i))
           plt.savefig(eigenfunplotfile.format(target,i), dpi=300)
           plt.close()
           if eigenfunh5file:
               with CheckpointFile(eigenfunh5file.format(target,i),'w') as afile:
                  afile.save_mesh(mesh)
                  afile.save_function(eigenfun)
           if i in plotefuns:
              eigenf_imgs.append(eigenfunplotfile.format(target,i))
           if pr <0.01:
              plt.clf()
      #        fig = plt.figure()
      #        axes= fig.add_subplot(projection="3d")
      #        collection=trisurf(eigenfun,axes=axes);
      #        collection.set_clim(-1.1,1.1)
      #        fig.colorbar(collection);
              fig, axes = plt.subplots()
              collection = tripcolor(eigenfun, axes=axes,cmap='bwr')
              collection.set_clim(-1.1,1.1)
              fig.colorbar(collection);
              plt.title('eigenfunction {}  $\lambda=${:7.5f} ratio {:1.5f}'.format( i, r.real,pr))
              PETSc.Sys.Print("> eigenfunction {} with p-ration{:1.5f} plotted to ".format(i,pr) + eigenfun_smpr_file.format(target,i))
              plt.savefig(eigenfun_smpr_file.format(target,i),dpi=300)
              plt.close()
              eigenf_imgs_smpr.append(eigenfun_smpr_file.format(target,i))
#    np.savetxt(eigenvalfile.format(target), eigenvalues)
#    print("> eigenvalues written to {}".format(eigenvalfile.format(target)))
    dd,dd2=divmod(nreq,25)
    for i in range(dd):
       segment=list(range(25*i,25*i+25))
       i0=segment[0]
       iend=segment[-1]
       eigenf_imgs=[]
       for j in segment:
          eigenf_imgs.append(eigenfunplotfile.format(target,j))
       combine_images(columns=5, space=20, images=eigenf_imgs,file=eigenfunmontagefile.format(target,i0))
       PETSc.Sys.Print("> eigenfunction montage between {} and {} is  written to {}".format(i0,iend,eigenfunmontagefile.format(target,i0)))
    return modes, eigenvalues_v2, pratio, eigenf_imgs_smpr, targets

def  get_eigenpairs_v3(Eps,nreq,Bsc,V,L,plotefuns,eigenfunplotfile,eigenfunmontagefile,eigenfun_smpr_file,eigenfunh5file=[],mesh=[]):
   # get eigenpairs
    eigenvalues = []
    eigenvalues_v2 = []
    pratio=[]
    modes=[]
    eigenf_imgs = []
    eigenf_imgs_smpr = []
    for i in range(nreq):
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
           eigenvalues_v2.append(r.real)
           modes.append(i)
           f2=assemble(eigenfun**2*dx)
           f4=assemble(eigenfun**4*dx)
           pr=1/(L**2)*(f2**2)/f4
           pratio.append(pr)
           plt.clf()
           fig, axes = plt.subplots()
           collection = tripcolor(eigenfun, axes=axes)
           collection.set_clim(-1.1,1.1)
           fig.colorbar(collection);
           plt.title(' eigenfunction {}  $\lambda=${:7.5f}  ratio {:1.5f}'.format( i, r.real,pr))
           PETSc.Sys.Print("> eigenfunction {} plotted to ".format(i) + eigenfunplotfile.format(i))
           plt.savefig(eigenfunplotfile.format(i), dpi=300)
           plt.close()
           if eigenfunh5file:
               with CheckpointFile(eigenfunh5file.format(i),'w') as afile:
                  afile.save_mesh(mesh)
                  afile.save_function(eigenfun)
           if i in plotefuns:
              eigenf_imgs.append(eigenfunplotfile.format(i))
           if pr <0.05:
              plt.clf()
              fig = plt.figure()
              axes= fig.add_subplot(projection="3d")
              collection=trisurf(eigenfun,axes=axes);
              collection.set_clim(-1.1,1.1)
              fig.colorbar(collection);
              plt.title('eigenfunction {}  $\lambda=${:7.5f} ratio {:1.5f}'.format( i, r.real,pr))
              PETSc.Sys.Print("> eigenfunction {} with p-ration{:1.5f} plotted to ".format(i,pr) + eigenfun_smpr_file.format(i))
              plt.savefig(eigenfun_smpr_file.format(i),dpi=300)
              plt.close()
              eigenf_imgs_smpr.append(eigenfun_smpr_file.format(i))
#    np.savetxt(eigenvalfile.format(target), eigenvalues)
#    print("> eigenvalues written to {}".format(eigenvalfile.format(target)))
    dd,dd2=divmod(nreq,25)
    for i in range(dd):
       segment=list(range(25*i,25*i+25))
       i0=segment[0]
       iend=segment[-1]
       eigenf_imgs=[]
       for j in segment:
          eigenf_imgs.append(eigenfunplotfile.format(j))
       combine_images(columns=5, space=20, images=eigenf_imgs,file=eigenfunmontagefile.format(i0))
       PETSc.Sys.Print("> eigenfunction montage between {} and {} is  written to {}".format(i0,iend,eigenfunmontagefile.format(i0)))
    return modes, eigenvalues_v2, pratio, eigenf_imgs_smpr


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
