Solve 

-div A grad u = lambda u

where A(x) = 1/(1+ sum_{integer n: x0<= n <= x1}f(x-n-dn(w))). 

Tested with 3 cases: 

operator 1 : -div A grad u = lambda u
operator 2 : -div A grad u = lambda Au
operator 3 : -div  grad u = lambda Au

# eigenvalue problem

000000   [0,2\pi]   constant coefficient    periodic boundary   operator=1 

000001   [0,2\pi]   constant coefficient    Dirichlet boundary  operator=1 

000002   [0,10]     random displacement     Dirichlet boundary  operator=1 

000003   [0,100]    random displacement     Dirichlet boundary  operator=1 

000004   [0,200]    random displacement     Dirichlet boundary  operator=1 

000005   [0,10]     random displacement v2  Dirichlet boundary   operator=1 

000006   [0,100]    random displacement v2  Dirichlet boundary   operator=1 

000007   [0,200]    random displacement v2  Dirichlet boundary   operator=1 

000008   [0,400]    random displacement v2  Dirichlet boundary  operator=1 

000009   [0,400]    random displacement     Dirichlet boundary  operator=1 

000010   [0,200]    random displacement     Dirichlet boundary  operator=1 

000011   [0,10]     fixed displacement (1-periodic)   Dirichlet boundary  operator=1 

000012   [0,100]    fixed displacement (1-periodic)   Dirichlet boundary  operator=1 

000013   [0,10]     1 - 10 2pc constants (2-periodic)   Dirichlet boundary  operator=1 

000014   [0,10]     fixed displacement (1-periodic)   Dirichlet boundary 

000015   [0,200]    operator=1 random displacement     Dirichlet boundary   all first 1000 eigenvalues

000016   [0,200]    operator=2 random displacement     Dirichlet boundary   all first 1000 eigenvalues

000017   [0,200]    operator=3 random displacement     Dirichlet boundary   all first 500 eigenvalues

2). The piecewise constant coefficient with random i.i.d from [1,10]: 

000018   [0,400]    operator=1 random displacement     Dirichlet boundary   all first 1000 eigenvalues

000019   [0,400]    operator=2 random displacement     Dirichlet boundary   all first 1000 eigenvalues
