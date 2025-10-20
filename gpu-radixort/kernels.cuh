#ifndef RADIX_KERS 
#define RADIX_KERS

#define GET_BITS(a, h, iter) ((1<<h)-1) & (a>>(iter*h))

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

template<int Q, int H>
__global__ void
histogramKernel(uint32_t *arr,
                uint32_t *glbHist,
                size_t N,
                int bits_iter);
{
    __shared__ uint32_t histShr[H];
    uint32_t block_offset = blockdim.x*blockidx.x;
    uint32_t arr_idx = block_offset + g*blockdim.x + threadidx.x;
    uint32_t key_idx;
    for (int q = 0; q < Q; q++) {
        if (arr_idx<N) {
            key_idx = GET_BITS(arr[arr_idx], lgH, bits_iter);
            atomicAdd(&histShr[key_idx], 1);
        }
    }
    __syncthreads();
    uint32_t chunk = (H + blockdim.x - 1) / blockdim.x;
    copyFromShr2GlbMem<uint32_t, chunk>(block_offset, N, glbHist, histShr);
}



#endif