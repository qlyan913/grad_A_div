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

THe coefficient 
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


# coefficient piecewise constant random i.i.d from [1,10]: 

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
