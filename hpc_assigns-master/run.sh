#!/bin/bash

#PBS -q class
#PBS -l nodes=2:sixcore
#PBS -l walltime=00:10:00
#PBS -N datprefix
##PBS -m e
##PBS -M plavin13@gatech.edu

TERM=xterm-256color
MPIRUN=/usr/lib64/openmpi/bin/mpirun

echo TESTING

cd $HOME/hpc_assigns
for p in 6 12
do
    mpirun -np $p --hostfile $PBS_NODEFILE ./poly_eval -n 1000 -m 10
done

