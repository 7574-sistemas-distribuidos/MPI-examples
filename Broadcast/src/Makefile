EXECS=mpi_broadcast
MPICC?=mpic++

all: ${EXECS}

mpi_broadcast: mpi_broadcast.cpp
	${MPICC} -o mpi_broadcast mpi_broadcast.cpp

clean:
	rm -f ${EXECS}
