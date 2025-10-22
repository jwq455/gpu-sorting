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
    }
}

// !!! ASSUMES THAT len(d_out)> len(shmem_red) OTHERWISE IT FAILS
template<class T, uint32_t CHUNK>
__device__ inline void
copyFromShr2GlbMem( const uint32_t glb_offs
                  , const uint32_t N
                  , T* d_out
                  , volatile T* shmem_red
) {
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        uint32_t loc_ind = threadIdx.x + blockDim.x * i;
        uint32_t glb_ind = glb_offs + loc_ind;
        if (glb_ind < N) {
            T elm = const_cast<const T&>(shmem_red[loc_ind]);
            d_out[glb_ind] = elm;
        }
    }
    __syncthreads(); // leave this here at the end!
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
//            if (blockIdx.x==1) printf("KeyIdx: %d\n", key_idx);
            atomicAdd(&histShr[key_idx], 1);
        }
    }
    __syncthreads();
    if (threadIdx.x==1 && blockIdx.x==1) {
        printf("Block %d Shared Historgram:\n", blockIdx.x);
        for (int i = 0; i < H; i++) {
            printf("%d ", histShr[i]);
        }
        printf("\n");
    }
    copyFromShr2GlbMem<uint32_t, CHUNK>(blockIdx.x*H, blockDim.x*H, glbHist, histShr);
}



#endif
