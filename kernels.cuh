#ifndef RADIX_KERS 
#define RADIX_KERS

#include <cuda_runtime.h>

#define GET_BITS(a, mask, shift) ((1<<mask)-1) & (a>>(shift*mask))


template<int H, int CHUNK>
__device__ inline void
copyFromShr2Glb(const uint32_t glb_offset,
                const uint32_t size_glb,
                uint32_t* d_out,
                volatile uint32_t* shmem)
{
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        uint32_t loc_ind = threadIdx.x + H * i;
        uint32_t glb_ind = glb_offset + loc_ind;
        if (glb_ind < size_glb) {
            uint32_t elm = const_cast<const uint32_t&>(shmem[loc_ind]);
            d_out[glb_ind] = elm;
        }
    }
    __syncthreads();
}


template<int B, int Q, int lgH, int H, int CHUNK>
__global__ void
histogramKernel(uint32_t *arr,
                uint32_t *glbHist,
                size_t N,
                int bits_iter)
{
    __shared__ uint32_t histShr[H];
    // Initialize shared memory to zero
    // Needs to handle when H<B, so we don't go out of bounds in histShr!
    histShr[threadIdx.x] = 0;
    if (H-B>0 && threadIdx.x < (H-B)) histShr[threadIdx.x+B] = 0;

    uint32_t block_offset = blockDim.x * blockIdx.x * Q;
    uint32_t key_idx;
    for (int q = 0; q < Q; q++) {
        uint32_t arr_idx = block_offset + q * blockDim.x + threadIdx.x;
        if (arr_idx<N) {
            key_idx = GET_BITS(arr[arr_idx], lgH, bits_iter);
            atomicAdd(&histShr[key_idx], 1);
        }
    }
    __syncthreads();
    copyFromShr2Glb<H, CHUNK>(blockIdx.x*H, blockDim.x*H, glbHist, histShr);
}



#endif
