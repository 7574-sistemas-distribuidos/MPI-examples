EXECS=mpi_pi
MPICC?=mpicc

all: ${EXECS}

mpi_pi: mpi_pi.c
	${MPICC} -o mpi_pi mpi_pi.c -std=c99

clean:
	rm -f ${EXECS}
