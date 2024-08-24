Solve eigenvalue 
     - div (A \nabla u)=\lambda u
 on 2D square [0, L]x[0, L]x[0,L]

 Tested with 3 cases: 

operator 1 : -div A grad u = lambda u
operator 2 : -div A grad u = lambda Au
operator 3 : -div  grad u = lambda Au

To check the eigenfunctions, we solve eigenproblem with eigenvalue closest to:

target= 0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600,800,1000,1200,1500,2000,2200,2500,3000.

File names: 
target_00000_smpr.pvd : 21 eigenfunctions with small participation ratio (<0.05) closest to target number
target_00000_eigen.pvd: 21 eigenfunctions cloest to target value
coefficient.pvd:    coefficients A.
ratio_eigen_log.png:  participation  ratio vs eigenvalues.

# coefficient piecewise constant random i.i.d from [1,10]: 

-- operator 1 -- 
0000000   [0,10]x[0,10]x[0,10]   Dirichlet boundary
0000001   [0,100]x[0,100]x[0,100]   Dirichlet boundary
-- operator 2 --    
0000002   [0,100]x[0,100]x[0,100]   Dirichlet boundary
-- operator 3 --    
0000003   [0,100]x[0,100]x[0,100]   Dirichlet boundary