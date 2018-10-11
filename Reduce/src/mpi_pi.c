// Example taken from: https://github.com/kiwenlau/MPI_PI/blob/master/Montecarlo/mpi_pi.c 
// This program is to caculate PI using MPI
// The algorithm is based on Monte Carlo method. The Monte Carlo method randomly picks up a large number of points in a square. It only counts the ratio of pints in side the circule.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <stdint.h>

int main (int argc, char* argv[]) {    
    // Initialize MPI (particularly the amount of processes)
    MPI_Init(&argc, &argv);
    
    //Get process ID
    int rank = 0;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    //Get processes Number
    int size = 0;
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    srand((int)time(0));
    printf("[RANK %d] Starting simulation\n", rank);
    
    //Each process will caculate a part of the sum
    double pi = 0.0; 
    double x = 0.0;
    double y = 0.0;
    int64_t result = 0;
    int64_t N = 1E9;

    int64_t iterations = N / size;
    printf("[RANK %d] Number of iterations: %d\n", rank, iterations);
    for (int i = 0; i < iterations; ++i) {
        x = rand() / (RAND_MAX + 1.0);
        y = rand() / (RAND_MAX + 1.0);

        if (x*x + y*y < 1.0) {
            ++result;
        }
    }
    
    //Sum up all results
    int64_t sum = 0;
    MPI_Reduce(&result, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    

    printf("[RANK %d] Stopping simulation\n", rank);    
    if (rank == 0) {
        pi = 4 * sum / (double)N;
        printf("[RANK %d] Number of Processes: %d - PI value: %0.12f\n", rank, size, pi);

    }
    
    MPI_Finalize();
    return 0;
}
