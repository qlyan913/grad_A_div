Solve 

-div A grad u = lambda u

where A(x) = 1/(1+ sum_{integer n: x0<= n <= x1}f(x-n-dn(w))). 

Tested with 3 cases: 

operator 1 : -div A grad u = lambda u
operator 2 : -div A grad u = lambda Au
operator 3 : -div  grad u = lambda Au

data eigenfunction saved in subfolder h5_file/ ***.h5
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

2). The piecewise constant coefficient with random i.i.d from [1,10], Dirichlet boundary : 

000018   [0,400]    operator=1     all first 1000 eigenvalues
000019   [0,400]    operator=2     all first 1000 eigenvalues
000028   [0,200]    operator=2     all first 800 eigenvalues
000029   [0,100]    operator=2     all first 500 eigenvalues
000030   [0,800]    operator=2     all first 1700 eigenvalues

000031   [0,200]    operator=2  (nc = 100)  all first 800 eigenvalues
000032   [0,200]    operator=2  (nc = 40)  all first 800 eigenvalues
000033   [0,200]    operator=2  (nc = 20)  all first 800 eigenvalues
000061   [0,200]    operator=2  (nc = 4)  all first 800 eigenvalues
000063   [0,200]    operator=2  (nc = 4)  all first 100 eigenvalues a_1=2, a_2=10

000034   [0,200]    operator=2  (nc = 2)  all first 800 eigenvalues
000062   [0,200]    operator=2  (nc = 2)  all first 100 eigenvalues a_1=2, a_2=10,a_3=6,a_4=3
000035   [0,200]    operator=2  (nc = 1, i.e., constant)  all first 800 eigenvalues

3). Test over different range of pc random vairable.
000036   [0,200],  pc i.i.d from [1,5],   operator=2     all first 800 eigenvalues
000037   [0,200],  pc i.i.d from [1,20],   operator=2     all first 800 eigenvalues
000038   [0,200],  pc i.i.d from [1,50],   operator=2     all first 800 eigenvalues
000039   [0,200],  pc i.i.d from [1,100],   operator=2     all first 800 eigenvalues
000040   [0,200],  pc i.i.d from [1,200],   operator=2     all first 800 eigenvalues

000041   [0,400],  pc i.i.d from [1,5],   operator=2     all first 800 eigenvalues
000042   [0,400],  pc i.i.d from [1,20],   operator=2     all first 800 eigenvalues
000043   [0,400],  pc i.i.d from [1,50],   operator=2     all first 800 eigenvalues
000044   [0,400],  pc i.i.d from [1,100],   operator=2     all first 800 eigenvalues
000045   [0,400],  pc i.i.d from [1,200],   operator=2     all first 800 eigenvalues
000046   [0,400],  pc i.i.d from [1,400],   operator=2     all first 800 eigenvalues

000050   [0,200],  pc i.i.d from [10,20],   operator=2     all first 800 eigenvalues
000051   [0,200],  pc i.i.d from [10,50],   operator=2     all first 800 eigenvalues
000052   [0,200],  pc i.i.d from [10,100],   operator=2     all first 800 eigenvalues

000047   [0,400],  pc i.i.d from [10,20],   operator=2     all first 800 eigenvalues
000048   [0,400],  pc i.i.d from [10,50],   operator=2     all first 800 eigenvalues
000049   [0,400],  pc i.i.d from [10,100],   operator=2     all first 800 eigenvalues

4). coefficient A is landscape u^2
000053   [0,200], V from [0,100], operator =2,  all first 800 eigenvalues 
000054   [0,200], V from [0,1], operator =2,  all first 800 eigenvalues 
000055   [0,200], V from [1,10], operator =2,  all first 800 eigenvalues 
000057   [0,200], V from [0,20], operator =2,  all first 800 eigenvalues 
000060   [0,1000],V from [0,10], operator =2,  all first 1700 eigenvalues 

5). coefficient A is 1/V^2
000056   [0,200], V from [0,100], operator =2,  all first 800 eigenvalues, nelts=10*L, deg =5
000059   [0,200], V from [0,100], operator =2,  all first 800 eigenvalues, nelts=L, deg=5

000058   [0,200], V from [0,20], operator =2,  all first 800 eigenvalues , nelts=10*L, deg =5


Testing for  pc coefficients with i.i.d from [1,10], operator = 2,  Dirichlet boundary:
all first 300 eigenvalues:
000020   [0,400], nelts=10*L, deg =1
000021   [0,400], nelts=10*L, deg =5
000022   [0,400], nelts=L, deg =1
000023   [0,400], nelts=L, deg =5

all first 700 eigenvalues:
000024   [0,800], nelts=10*L, deg =1
000025   [0,800], nelts=10*L, deg =5
000026   [0,800], nelts=L, deg =1
000027   [0,800], nelts=L, deg =5



Two alternative constants a1=1, a2=10, 
000070 L=20, p ratio is periodic in T=20 as expected