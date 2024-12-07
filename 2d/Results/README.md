Solve eigenvalue 
     - div (A \nabla u)=\lambda u
 on 2D square [0, L]x[0, L].

 Tested with 3 cases: 

operator 1 : -div A grad u = lambda u
operator 2 : -div A grad u = lambda Au
operator 3 : -div  grad u = lambda Au

To check the eigenfunctions, we solve eigenproblem with eigenvalue closest to:

target= 0,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600,800,1000,1200,1500,2000,2200,2500,3000.

File names: 
eigenfunmontage_smpr_ n.png :    saved  montage of eigenfunctions with small participation ratio (<0.05)
coefficient.pn:    coefficients A.
target_a_smpr_b.png:  b th eigenfunctions whose eigenvalues cloeset to number a and with small participation ratio (<0.05)  
pratio_eigen_log.pn:  participation  ratio vs eigenvalues.
eigen_pratio.csv: eigenvalues and participation ratio.

1). The random displacement coefficient 
 A(x) = 1/(1+ sum_{integer n: x0<= n <= x1}f(x-n-dn(w)))

Choices of f:
(a). f1 = 20x[max{(1-x^2/s^2)^3,0}(3x^2+1)], supp(f) in [-s,s]
(b). f2 =(-1)^(i+j) x 0.75 [max{(1-x^2/s^2)^3,0}(3x^2+1)], supp(f) in [-s,s]
(c). f3 =  -0.9x[max{(1-x^2/s^2)^3,0}(3x^2+1)] when (-1)^(i+j)  == -1
        =  20 x[max{(1-x^2/s^2)^3,0}(3x^2+1)] when (-1)^(i+j)  == 1

# eigenvalue problem

000000   [0,1]x[0,1]   constant coefficient    Dirichlet boundary 

# coefficient f1:
-- operator 1 --
000001   [0,10]x[0,10]   fixed displacement    Dirichlet boundary
000002   [0,10]x[0,10]   random displacement    Dirichlet boundary 
000003   [0,100]x[0,100]   random displacement   Dirichlet boundary 
-- operator 2 --
000004   [0,100]x[0,100]   random displacement   Dirichlet boundary
-- operator 3 --
000005   [0,100]x[0,100]   random displacement   Dirichlet boundary

# coefficient f3:  
-- operator 1 -- 
000006   [0,10]x[0,10]   random displacement   Dirichlet boundary
000007   [0,100]x[0,100]   random displacement   Dirichlet boundary
-- operator 2 --  
000008   [0,100]x[0,100]   random displacement   Dirichlet boundary 
-- operator 3 --
000009   [0,100]x[0,100]   random displacement   Dirichlet boundary   


2). The piecewise constant coefficient with random i.i.d from [1,10]: 

-- operator 1 -- 
0000010   [0,10]x[0,10]   Dirichlet boundary
0000011   [0,200]x[0,200]   Dirichlet boundary
0000014   [0,500]x[0,500]  Dirichlet boundary
-- operator 2 --    
0000012   [0,200]x[0,200]   Dirichlet boundary
0000015   [0,500]x[0,500]  Dirichlet boundary
-- operator 3 --    
0000013   [0,200]x[0,200]   Dirichlet boundary
0000016   [0,500]x[0,500]  Dirichlet boundary

0000017   [0,50]x[0,50]  (nc=2*2) Dirichlet boundary first 4000 eigenfunctions
0000018   [0,50]x[0,50]  (nc=50*50) Dirichlet boundary first 4000 eigenfunctions
0000019   [0,100]x[0,100] nc=4*4 Dirichlet boundary first 2000 eigenfunctions
0000020   [0,100]x[0,100] nc=100*100 Dirichlet boundary first 5000 eigenfunctions


spectrum slicing test:
L=100, deg=5, nc =5*5, 
1. interval [0,0.5], np=1, time: 136 
2. interval [0,1], np=1, time:265


0000021 [0,100]x[0,100] nc=5*5 Dirichlet boundary, first 1200 eigenfunctions 
0000022 [0,100]x[0,100] nc=5*5 Dirichlet boundary, spectrum slicing on [0,1], np=1 
0000023 [0,100]x[0,100] nc=5*5 Dirichlet boundary, spectrum slicing on [1,2], np=1 
0000024 [0,100]x[0,100] nc=5*5 Dirichlet boundary, spectrum slicing on [2,3], np=1 

0000026 [0,100]x[0,100] nc=100*100 Dirichlet boundary, spectrum slicing on [0,1], np=1 
0000027 [0,100]x[0,100] nc=100*100 Dirichlet boundary, spectrum slicing on [1,2], np=1 
0000028 [0,100]x[0,100] nc=100*100 Dirichlet boundary, spectrum slicing on [2,3], np=1 
0000029 [0,100]x[0,100] nc=100*100 Dirichlet boundary, spectrum slicing on [3,4], np=1 

Target eigenvalues = np.linspace(0,1000,5001)
0000030 [0,100]x[0,100] nc=100*100 Dirichlet boundary
0000031 [0,100]x[0,100] nc=10*10 Dirichlet boundary

MPI plot:
1. create_coef.py generate coefficient function and save it to coef.h5.  
2. main_pc_mpi.py save eigenfunction to .h5
3. plot_mpi.py load eigen.h5 and plot function

0000025 [0,100]x[0,100] nc=2*2 Dirichlet boundary, first 101 eigenfunctions 



