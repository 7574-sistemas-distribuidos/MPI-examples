EXECS=mpi_scatter
MPICC?=mpic++

all: ${EXECS}

mpi_scatter: mpi_scatter.cpp
	${MPICC} -o mpi_scatter mpi_scatter.cpp

clean:
	rm -f ${EXECS}
