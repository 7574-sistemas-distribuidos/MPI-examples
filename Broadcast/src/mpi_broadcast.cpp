#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <stdint.h>

template <int32_t SIZE>
void data_generation(int32_t rank, double* & rand_array) {
    if (rank == 0) {
        for (int32_t i = 0; i < SIZE; ++i) {
            rand_array[i] = rand() / (RAND_MAX + 1.0); 
        }
    }
}

template <int32_t SIZE>
int32_t dart_computation(int32_t rank, int32_t processes, const double* rand_array) {
    int32_t data_size = SIZE / processes;
    int32_t begin = (rank-1) * data_size;
    int32_t end = (rank-1) * data_size + data_size;

    int32_t result = 0;
    printf("[RANK %d] Dart computation. Begin: %d - End: %d\n", rank, begin, end-1);

    double x = 0.0;
    double y = 0.0;
    for (int32_t i = begin; i < end; i+=2) {
        x = rand_array[i];
        y = rand_array[i+1];
        if (x*x + y*y < 1.0) {
            ++result;
        }
    }
    return result;
}


int main (int argc, char* argv[]) {    
    // Initialize MPI (particularly the amount of processes)
    MPI_Init(&argc, &argv);
    
    //Get process ID
    int rank = 0;
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    //Get processes Number
    int size = 0;
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    // Number of processes doing some computation
    int processes = size - 1;

    // Start 
    srand((int)time(0));
    printf("[RANK %d] Starting simulation\n", rank);

    // Generate a huge array of random numbers JUST in rank 0 process
    // MAX_INT:  2147483647
    const int32_t N = 50000000;
    double* rand_array = new double[N];
    data_generation<N>(rank, rand_array);

    if (rank == 0) {
        printf("[RANK %d] Proceed to send data\n", rank);
        MPI_Bcast(rand_array, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        printf("[RANK %d] Data has been sent\n", rank);

        int32_t cum_results = 0;
        int32_t remote_result = 0;
        // Receive the hits from all the processes
        for (int32_t i = 1; i < size; ++i) {
            MPI_Recv(&remote_result, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("[RANK %d] Result received from Process %d. Result: %d\n", rank, i, remote_result);
            cum_results += remote_result;
        }

        double pi_estimation = 4 * cum_results / (double) (N/2);
        printf("[RANK %d] All results have arrived. Pi estmation: %0.12f", pi_estimation);
    }
    else {
        printf("[RANK %d]: Proceed to receive data\n", rank);
        MPI_Bcast(rand_array, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        printf("[RANK %d] Data has been received\n", rank);

        // Execute the computation
        int32_t result = dart_computation<N>(rank, processes, rand_array);
        printf("[RANK %d] Proceed to send result to root: %d\n", rank, result);
        MPI_Send(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    delete [] rand_array;
    MPI_Finalize();
    return 0;
}
