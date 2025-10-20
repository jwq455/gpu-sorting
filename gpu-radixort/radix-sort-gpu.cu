#include "../helper.h"
#include "kernels.cuh"

using namespace std;

#define GPU_RUNS    50

template<int B, int Q, int lgH>
void radixSort(uint32_t *d_A, uint32_t *d_B, uint32_t *h_B, size_t N) {
    unsigned long elementsPerBlock = B*Q;
    // Setup execution parameters
    const int blocks = (N + elementsPerBlock - 1) / elementsPerBlock;
    const int H = 1<<lgH;
    const int CHUNK = (H + B - 1) / B; 

    // Temporary I/O buffers
    // we use d_A - will be overwritten, is this ok??

    // uint32_t *d_ind;
    // cudaMalloc((void **) &d_ind, N);
    // cudaMemcpy(d_ind, d_A, N, cudaMemcpyDeviceToDevice);

    // We use d_B

    // uint32_t *d_out;
    // cudaMalloc((void **) &d_out, N);
    // cudaMemset(d_out, 0, N);

    // global Historgram buffer
    uint32_t *glbHist;
    cudaMalloc((void **) &glbHist, blocks*H);
    cudaMemset(glbHist, 0, blocks*H);

    // Loop over sizeof(elem)/lgH
    // for (int i_cpu = 0; i_cpu < sizeof(uint32_t)/lgH; i_cpu++) {
        // globla_hist[blocks][H]
    histogramKernel<Q, H, CHUNK><<<blocks, B>>>(d_A, glbHist, N, i_cpu);
        // Pseudo - use kernels from assignments
        // transpose_hist()
        // scan_hist()
        // transpose_scan()
        // Second kernel - Does sorting and scattering into global memory

        // Update d_ind = d_out 

    // }

    uint32_t *hist_h = (uint32_t *)malloc(sizeof(uint32_t)*blocks*H);
    cudaMemcpy(hist_h, glbHist, blocks*H, cudaMemcpyDeviceToHost);

    for (int i = 0; i < blocks; i++) {
        for (int j = 0; j < H; j++) {
            printf("%d ", hist_h[i*blocks + j]);
        }
        printf("\n"); 
    }
    
}

template<int B, int Q, int lgH>
void runRadixSort(uint32_t *d_A, uint32_t *d_B, uint32_t *h_B, size_t N) {
    // dry run
    radixSort<B, Q, lgH>(d_A, d_B, h_B, N);
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
    uint32_t *h_A = (uint32_t*)calloc(N, sizeof(uint32_t));
    uint32_t *h_B = (uint32_t*)calloc(N, sizeof(uint32_t));

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