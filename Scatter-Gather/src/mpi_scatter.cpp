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

int32_t dart_computation(const double* rand_array, const int32_t size) {
    int32_t result = 0;
    double x = 0.0;
    double y = 0.0;

    for (int32_t i = 0; i < size; i+=2) {
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
    const int32_t N = 10000000;
    const int32_t N_PER_PROCESS = N / processes;
    double* rand_array = NULL;
    double* scatter_rand_array = NULL;
    scatter_rand_array = new double[N_PER_PROCESS];

    if (rank == 0) {
        rand_array = new double[N];
        data_generation<N>(rank, rand_array);

        // Scatter the samples
        printf("[RANK %d] Proceed to send data\n", rank);
        MPI_Scatter(rand_array, N_PER_PROCESS, MPI_DOUBLE, scatter_rand_array, N_PER_PROCESS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        printf("[RANK %d] Data has been sent\n", rank);

        // Gather the results
        int32_t results[processes];
        int32_t cum_results = 0;

        // recv_counter is the amount of metrics received PER PROCESS
        MPI_Gather(&results, 1, MPI_INT, &results, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Receive the hits from all the processes
        for (int32_t i = 1; i < size; ++i) {
            printf("[RANK %d] Result received from Process %d. Result: %d\n", rank, i, results[i]);
            cum_results += results[i];
        }

        double pi_estimation = 4 * cum_results / (double) (N/2);
        printf("[RANK %d] All results have arrived. Pi estmation: %0.12f", pi_estimation);

        delete [] rand_array;
    }
    else {
        // scatter_rand_array = new double[N_PER_PROCESS];
        printf("[RANK %d] Proceed to receive data\n", rank);
        MPI_Scatter(rand_array, N_PER_PROCESS, MPI_DOUBLE, scatter_rand_array, N_PER_PROCESS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        printf("[RANK %d] Data has been received\n", rank);

        // Execute the computation
        int32_t result = dart_computation(scatter_rand_array, N_PER_PROCESS);

        // Send the results to the ROOT process via a Gather
        printf("[RANK %d] Proceed to send result to root: %d\n", rank, result);
        MPI_Gather(&result, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    delete [] scatter_rand_array;
    MPI_Finalize();
    return 0;
}
