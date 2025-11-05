#include "../helper.h"
#include "kernels.cuh"

using namespace std;

// FROM ASSIGNMENT 2
uint32_t nextMul32(uint32_t x) {
    return ((x + 31) / 32) * 32;
}

/**
 * FROM ASSIGNMENT 2 - NUMBER OF BLOCKS FOR SCAN!
 * `N` is the input-array length
 * `B` is the CUDA block size
 * This function attempts to virtualize the computation so
 *   that it spawns at most 1024 CUDA blocks; otherwise an
 *   error is thrown. It should not throw an error for any
 *   B >= 64.
 * The return is the number of blocks, and `CHUNK * (*num_chunks)`
 *   is the number of elements to be processed sequentially by
 *   each thread so that the number of blocks is <= 1024.
 */
template<int CHUNK>
uint32_t getNumBlocks(const uint32_t N, const uint32_t B, uint32_t* num_chunks) {
    const uint32_t max_inp_thds = (N + CHUNK - 1) / CHUNK;
    const uint32_t num_thds0    = min(max_inp_thds, MAX_HWDTH);

    const uint32_t min_elms_all_thds = num_thds0 * CHUNK;
    *num_chunks = max(1, (N + min_elms_all_thds - 1) / min_elms_all_thds);

    const uint32_t seq_chunk = (*num_chunks) * CHUNK;
    const uint32_t num_thds = (N + seq_chunk - 1) / seq_chunk;
    const uint32_t num_blocks = (num_thds + B - 1) / B;

    if(num_blocks <= MAX_BLOCK) {
        return num_blocks;
    } else {
        //printf("Warning: reduce/scan configuration does not allow the maximal concurrency supported by hardware.\n");
        const uint32_t num_blocks = 1024;
        const uint32_t num_thds   = num_blocks * B;
        const uint32_t num_conc_elems = num_thds * CHUNK;
        *num_chunks = (N + num_conc_elems - 1) / num_conc_elems;
        return num_blocks;
    }
}

template<int B, int Q, int lgH>
void radixSort(uint32_t *d_A, uint32_t *d_B, uint32_t *h_B, uint32_t *hist, uint32_t *hist_tr, uint32_t *hist_scan, uint32_t *hist_scan_tr,   size_t N) {
    unsigned long elementsPerBlock = B*Q;
    // Setup execution parameters

    // For histogram kernel
    const int blocks = (N + elementsPerBlock - 1) / elementsPerBlock;
    const int H = 1<<lgH;
    const int passes = (sizeof(uint32_t)*8 + lgH-1)/lgH;
    int hist_size = blocks*H;

    // For transpose kernel
    int  dimy = (blocks+TILE_SIZE-1) / TILE_SIZE;
    int  dimx = (H+TILE_SIZE-1) / TILE_SIZE;

    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid (dimx, dimy, 1);
    dim3 grid2 (dimy, dimx, 1);

    // For scan kernel
    // COPIED from scaninc() in host_skel.cuh assignment-2
    const uint32_t tp_size = sizeof(uint32_t);
    const uint32_t CHUNK_SCAN = ELEMS_PER_THREAD*4 / tp_size;
    uint32_t num_seq_chunks;
    uint32_t num_blocks = getNumBlocks<CHUNK_SCAN>(hist_size, B, &num_seq_chunks);
    const size_t   shmem_size = B * tp_size * CHUNK_SCAN;

    //

    // Meassuring performance should not contain all the memory allocation
    // you need three buffers for the input/output -> first iter reads from d_A, writes to d_B, then write d_b into d_ind (third iteration)
    uint32_t *tmp_inp;
    uint32_t *tmp_out = d_B;
    cudaMalloc((void **) &tmp_inp, sizeof(uint32_t)*N);
    cudaMemcpy(tmp_inp, d_A, sizeof(uint32_t)*N, cudaMemcpyDeviceToDevice);
    uint32_t *sort_mem_ptr = tmp_inp;
	
    uint32_t* d_tmp;
    cudaMalloc((void**)&d_tmp, MAX_BLOCK*sizeof(uint32_t));
    // Loop over sizeof(elem)/lgH
    for (int i_cpu = 0; i_cpu < passes; i_cpu++) {

        int bits = ((32 - i_cpu*lgH) >= lgH) ? lgH : (32 - i_cpu*lgH);
        const int H_curr = 1<<bits;
        hist_size = blocks*H_curr;


        // globla_hist[blocks][H]
        histogramKernel<B, Q, H, lgH><<<blocks, B, H_curr*sizeof(uint32_t)>>>(sort_mem_ptr, hist, N, i_cpu, bits);

        dimx = (H_curr+TILE_SIZE-1) / TILE_SIZE;
        dim3 grid (dimx, dimy, 1);
        dim3 grid2 (dimy, dimx, 1);

        // tanspose
        coalsTransposeKer<uint32_t,TILE_SIZE> <<<grid, block>>>
                        (hist, hist_tr, blocks, H_curr, H);

        num_blocks = getNumBlocks<CHUNK_SCAN>(hist_size, B, &num_seq_chunks); 
        // scan
        {
            redAssocKernel<Add<uint32_t>, CHUNK_SCAN><<< num_blocks, B, shmem_size >>>(d_tmp,  hist_tr, hist_size, num_seq_chunks);

            {
                const uint32_t block_size = nextMul32(num_blocks);
                const size_t shmem_size = block_size * sizeof(uint32_t);
                scan1Block<Add<uint32_t>><<< 1, block_size, shmem_size>>>(d_tmp,  num_blocks);
            }

            scan3rdKernel<Add<uint32_t>, CHUNK_SCAN><<< num_blocks, B, shmem_size >>>(hist_scan, hist_tr,d_tmp, hist_size, num_seq_chunks);
        }

        // transpose
        coalsTransposeKer<uint32_t,TILE_SIZE> <<<grid2, block>>>
                        (hist_scan, hist_scan_tr, H_curr, blocks, blocks);


        // Second kernel - Does sorting and scattering into global memory
        partitionScatterKer<B, Q, lgH><<<blocks, B, sizeof(uint32_t)*B*Q+H_curr*sizeof(uint32_t)*2>>>(sort_mem_ptr, N, hist, hist_scan_tr, tmp_out, i_cpu, bits);
        
	
	// Update d_ind = d_out
        // pointer swap - DON'T MEMCPY
        tmp_inp = sort_mem_ptr;
        sort_mem_ptr = tmp_out;
        tmp_out = tmp_inp;
    }

    cudaMemcpy(d_B, sort_mem_ptr, sizeof(uint32_t)*N, cudaMemcpyDeviceToDevice);

}

template<int B, int Q, int lgH>
void runRadixSort(uint32_t *d_A, uint32_t *d_B, uint32_t *h_B, uint32_t *hist, uint32_t *hist_tr, uint32_t *hist_scan, uint32_t *hist_scan_tr,   size_t N) {
    // dry run
    radixSort<B, Q, lgH>(d_A, d_B, h_B, hist, hist_tr, hist_scan, hist_scan_tr,   N);
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );

    //Get runtime
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int i=0; i<GPU_RUNS; i++) {
        radixSort<B, Q, lgH>(d_A, d_B, h_B, hist, hist_tr, hist_scan, hist_scan_tr,   N);
    }
    cudaDeviceSynchronize();
    gpuAssert( cudaPeekAtLastError() );


    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    // // CHECK MEMORY BOUND PERFORMANCE ANALYSIS!

    // gpuAssert( cudaPeekAtLastError() );
    printf("Radix Sorting for N=%lu runs in: %.2f us, Sorted keys per second: %.2f\n", N, elapsed, (N/(elapsed/1e6)));

    // // Print and validate :)
    printf("Validating result... ");
    cudaMemcpy(h_B, d_B, sizeof(uint32_t)*N, cudaMemcpyDeviceToHost);
    validate<uint32_t>(h_B, N);

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
    cudaMalloc((void **) &d_A, sizeof(uint32_t)*N);
    cudaMalloc((void **) &d_B, sizeof(uint32_t)*N);
    
    	//And for histograms
    unsigned long elementsPerBlock = B*Q;
    const int blocks = (N + elementsPerBlock - 1) / elementsPerBlock;
    const int H = 1<<lgH;
    int hist_size = blocks*H;
    const int hist_mem_size = sizeof(uint32_t)*hist_size;

    uint32_t *hist;
    uint32_t *hist_tr;
    uint32_t *hist_scan;
    uint32_t *hist_scan_tr;
    cudaMalloc((void **) &hist, hist_mem_size);
    cudaMalloc((void **) &hist_tr, hist_mem_size);
    cudaMalloc((void **) &hist_scan, hist_mem_size);
    cudaMalloc((void **) &hist_scan_tr, hist_mem_size);

    
    // Copy host memory to device
    cudaMemcpy(d_A, h_A, sizeof(uint32_t)*N, cudaMemcpyHostToDevice);

    printf("Size of A: %d\n", N);

    // compute efficient radix sort (validation and timing is done in runRadixSort())
    {
        runRadixSort<B, Q, lgH>(d_A, d_B, h_B, hist, hist_tr, hist_scan, hist_scan_tr, N);
    }

    //clean memory
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(hist);
    cudaFree(hist_tr);
    cudaFree(hist_scan);
    cudaFree(hist_scan_tr);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s size-A\n", argv[0]);
        exit(1);
    }

    cudaSetDevice(1);
    initHwd();

    const size_t SIZE_A = atoi(argv[1]);
				//Passed from makefile 
    const int B     = B_def; // Thread-block size
    const int Q     = Q_def;  // Number of elements processed by each thread
    const int lgH   = lgH_def; // Number of bits processed in each pass of counting sort

    printf("Running with parameters B: %d, Q: %d, and lgH; %d\n", B, Q, lgH);
    runAll<B, Q, lgH>(SIZE_A);

    return 0;
}

