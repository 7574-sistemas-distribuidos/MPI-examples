#!/bin/bash
export ROOTPATH="$(dirname "$(readlink -f "$0")")"

usage() {
    echo "Usage:"
    echo -e "\t./run_container.sh [MPI_PROCESSORS_NUMBER]"
    exit 1
}

# Script receives two parameters
# 1: Amount of Processors to run in the MPI simulation
if [ "$#" -ne "1" ] ; then
    usage
fi

# Build the example
docker build -f $(pwd)/Dockerfile -t openmpi_example:v0.0.1 .

# Run the example
docker run --privileged=true --rm -e PROCESSORS="$1" -it openmpi_example:v0.0.1 
