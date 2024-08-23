Solve eigenvalue 
     - div (A \nabla u)=\lambda u
 on 2D square [0, L]x[0, L].
THe coefficient 
 A(x) = 1/(1+ sum_{integer n: x0<= n <= x1}f(x-n-dn(w)))

Choices of f:
(a). f1 = 20x[max{(1-x^2/s^2)^3,0}(3x^2+1)], supp(f) in [-s,s]
(b). f2 =(-1)^(i+j) x 0.75 [max{(1-x^2/s^2)^3,0}(3x^2+1)], supp(f) in [-s,s]
(c). f3 =  -0.9x[max{(1-x^2/s^2)^3,0}(3x^2+1)] when (-1)^(i+j)  == -1
        =  20 x[max{(1-x^2/s^2)^3,0}(3x^2+1)] when (-1)^(i+j)  == 1

# eigenvalue problem

000000   [0,1]x[0,1]   constant coefficient    Dirichlet boundary 

# coefficient 1:
-- operator 1 --
000001   [0,10]x[0,10]   fixed displacement    Dirichlet boundary
000002   [0,10]x[0,10]   random displacement    Dirichlet boundary 
000003   [0,100]x[0,100]   random displacement   Dirichlet boundary 
-- operator 2 --
000004   [0,100]x[0,100]   random displacement   Dirichlet boundary
-- operator 3 --
000005   [0,100]x[0,100]   random displacement   Dirichlet boundary

# coefficient 3:  
-- operator 1 -- 
000006   [0,10]x[0,10]   random displacement   Dirichlet boundary
000007   [0,100]x[0,100]   random displacement   Dirichlet boundary
-- operator 2 --  
000008   [0,100]x[0,100]   random displacement   Dirichlet boundary 
-- operator 3 --
000009   [0,100]x[0,100]   random displacement   Dirichlet boundary   

# coefficient piecewise constant: 
-- operator 1 -- 
0000010   [0,10]x[0,10]   Dirichlet boundary
0000011   [0,200]x[0,200]   Dirichlet boundary
-- operator 2 --    
0000012   [0,200]x[0,200]   Dirichlet boundary
-- operator 3 --    
0000013   [0,200]x[0,200]   Dirichlet boundary