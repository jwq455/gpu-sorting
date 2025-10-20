#include "../helper.h"
#include "kernels.cuh"

using namespace std;

#define GPU_RUNS    50

runRadixSort<B, Q, lgH>(d_A, d_B, h_B, N) {
    unsigned long elementsPerBlock = B*Q;
    cd 
    // Setup execution parameters
    const int grid = (N + elementsPerBlock - 1) / elementsPerBlock;
    
    // dry run
    // RUN ALGORITHM HERE!
    histogramKernel<<<grid, B>>>()
    transposeKer()
    scanKernel()


    runRadixKernel<lgH><<<grid, B>>>(d_X, d_X_tr, M, N);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );


    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int i=0; i<GPU_RUNS; i++) {
        // IMPLEMENT RADIX SORT!

    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS;

    // CHANGE TO MEMORY BOUND PERFORMANCE ANALYSIS
    double  microsecPerSort = elapsed;
    double flopsPerMatrixMul = 3.0 * M * K * K * N;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-3f) / microsecPerSort>

    gpuAssert( cudaPeekAtLastError() );

    // Print and validate :)
    
}

template<int B, int Q, int lgH>
void runAll(size_t N) {
    srand(2025); 

    // Allocate host memory for input and output array
    uint32_t *h_A = (uint32_t*)calloc(sizeof(uint32_t), N);
    uint32_t *h_B = (uint32_t*)calloc(sizeof(uint32_t), N);

    // Initialize input array
    randomInit<uint32_t>(h_A, N);

    // Allocate device memory
    uint32_t *d_A;
    uint32_t *d_B;
    cudaMalloc((void **) &d_A, N);
    cudaMalloc((void **) &d_B, N);

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, N, cudaMemcpyHostToDevice);

    printf("Size of A: %d\n", N);

    // compute efficient radix sort (validation and timing is done in runRadixSort())
    {
        runRadixSort<B, Q, lgH>(d_A, d_B, h_B, N);
    }

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s size-A\n", argv[0]);
        exit(1);
    }
    const size_t SIZE_A = atoi(argv[1]);

    const int B     = 256; // Thread-block size
    const int Q     = 22;  // Number of elements processed by each thread
    const int lgH   = 8;   // Number of bits processed in each pass of counting sort

    runAll<B, Q, lgH>(SIZE_A);

    return 0;
}