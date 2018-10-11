#!/bin/bash
export ROOTPATH="$(dirname "$(readlink -f "$0")")"

# Compile the example
. /opt/env/openmpi-3.0.0.sh
cd $ROOTPATH
make -f $ROOTPATH/Makefile

# Execute the example
time mpirun --hostfile $ROOTPATH/hostfile --allow-run-as-root -np $PROCESSORS $ROOTPATH/mpi_scatter
