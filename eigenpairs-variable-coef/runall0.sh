#!/bin/bash

# set run parameters

coef=uniform
nc=1024
seed=13
amin=1.0
amax=100.0
params="$nc $amin $amax $seed"  # if no seed, different random numbers each time
deg=8
npts=6001
nelts=$nc
x0=0.
x1=$nc
bc=neumann
saveefuns=100,999
plotefuns=0,10,20,50,100,200,300,400,500,600,700,800,900,999
nreq=1000
target=0. # get eigenvalues nearest this target

# run program to compute and save data, capture output since 1st line gives output directory
output=`mktemp`
echo ./eigenpairs-vc.py --bc $bc --nelts $nelts --x0 " $x0" --x1 " $x1" --deg $deg --coef $coef --npts $npts --params $params --nreq $nreq --target $target --saveefuns $saveefuns --plotefuns $plotefuns --verbose
./eigenpairs-vc.py --bc $bc --nelts $nelts --x0 " $x0" --x1 " $x1" --deg $deg --coef $coef --npts $npts --params $params --nreq $nreq --target $target --saveefuns $saveefuns --plotefuns $plotefuns --verbose | tee $output
ddir=`head -1 $output`
/bin/rm $output

