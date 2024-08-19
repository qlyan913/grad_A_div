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
000002   [0,10]x[0,10]   random displacement    Dirichlet boundar 
000003   [0,100]x[0,100]   random displacement   Dirichlet boundar   first 300 eigen
000004   [0,100]x[0,100]   random displacement   Dirichlet boundar  300 eigens close to 20, p ratio aorund 0.05


# coefficient 2:    all p ratio lager than 0.3
-- operator 1 -- 
000005   [0,10]x[0,10]   random displacement   Dirichlet boundar   
000006   [0,100]x[0,100]   random displacement   Dirichlet boundar first 300 eigen
000007   [0,100]x[0,100]   random displacement   Dirichlet boundar 300 eigens close to 20

# coefficient 3:
-- operator 1 -- 
00008   [0,10]x[0,10]   random displacement   Dirichlet boundar,   
00009   [0,10]x[0,10]   random displacement   Dirichlet boundar   close to 20,   some p ratio less than 0.03
000010   [0,100]x[0,100]   random displacement   Dirichlet boundar first 300 eigen 
000011   [0,100]x[0,100]   random displacement   Dirichlet boundar 300 eigen close to 20, p-ratio  around 0.06
