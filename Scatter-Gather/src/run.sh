#!/bin/bash
export ROOTPATH="$(dirname "$(readlink -f "$0")")"

# Compile the example
cd $ROOTPATH
make -f $ROOTPATH/Makefile

# Execute the example
time mpirun --hostfile $ROOTPATH/hostfile --allow-run-as-root -np $PROCESSORS $ROOTPATH/mpi_scatter
