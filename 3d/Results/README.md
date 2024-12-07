Solve eigenvalue 
     - div (A \nabla u)=\lambda u
 on 2D square [0, L]x[0, L]x[0,L]

 Tested with 3 cases: 

operator 1 : -div A grad u = lambda u
operator 2 : -div A grad u = lambda Au
operator 3 : -div  grad u = lambda Au

To check the eigenfunctions, we solve eigenproblem with eigenvalue closest to:

target = [0,10,20,40,60,80,100,200,250,300,400,500,600,800,1000], 
         [1200,1500, 2000,2500,3000,3500,4000]
         [4500,5000,5500,6000,6500,7000,7500,8000]

File names: 
target_00000_smpr.pvd : 21 eigenfunctions with small participation ratio (<0.05) closest to target number
target_00000_eigen.pvd: 21 eigenfunctions cloest to target value
coefficient.pvd:    coefficients A.
ratio_eigen_log.png:  participation  ratio vs eigenvalues.
eigen_pratio.csv: eigenvalues and participation ratio.
eigen_pratio_1.csv: target=[0,10,20,40,60,80,100,200,250,300,400,500,600,800,1000], 
eigen_pratio_2.csv: target=[1200,1500, 2000,2500,3000,3500,4000], 
eigen_pratio_3.csv: target= [4500,5000,5500,6000,6500,7000,7500,8000]

1). The piecewise constant coefficient with random i.i.d from [1,10]: 

-- operator 1 -- 
0000000   [0,10]x[0,10]x[0,10]   Dirichlet boundary,
0000001   [0,50]x[0,50]x[0,50]   Dirichlet boundary.
-- operator 2 --    
0000002   [0,50]x[0,50]x[0,50]   Dirichlet boundary
-- operator 3 --    
0000003   [0,50]x[0,50]x[0,50]   Dirichlet boundary