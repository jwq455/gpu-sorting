#ifndef RADIX_KERS
#define RADIX_KERS

#include <cuda_runtime.h>
#include "pbb_kernels.cuh"
#include "../helper.h"

// #define GET_BITS(a, mask, shift) ((1<<mask)-1) & (a>>(shift*mask))
#define GET_BITS(a, mask, shift, lgH) ((1<<mask)-1) & (a>>(shift*lgH))
// Gets the i*lgH+k'th bit of 'a'
#define GET_KTH_BIT_UNSET(a, i, lgH, k) (1) ^ ((1) & (a>>(i*lgH+k))) 


template<int H, int B>
__device__ inline void
copyFromShr2Glb(const uint32_t glb_offset,
                const uint32_t size_glb,
                const uint32_t size_loc,
                uint32_t* d_out,
                volatile uint32_t* shmem)
{
    #pragma unroll
    for (uint32_t i = 0; threadIdx.x + B * i < size_loc; i++) {
        uint32_t loc_ind = threadIdx.x + B * i;
        uint32_t glb_ind = glb_offset + loc_ind;
        if (glb_ind < size_glb) {
            uint32_t elm = const_cast<const uint32_t&>(shmem[loc_ind]);
            d_out[glb_ind] = elm;
        }
    }
    __syncthreads();
}

template<int B, int Q>
__device__ inline void
copyFromGlb2Shr(const uint32_t glb_offset,
                const uint32_t N,
                uint32_t* d_ind,
                volatile uint32_t* shrmem)
{
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        uint32_t loc_ind = i * B + threadIdx.x;
        uint32_t glb_ind = glb_offset + loc_ind;
        if (glb_ind<N) {
            shrmem[loc_ind] = d_ind[glb_ind];
        }
    }
}

template<int Q>
__device__ inline void
copyFromShr2Reg(uint32_t *regElem, 
                volatile uint32_t *shrmem)
{
    #pragma unroll
    for (int i = 0; i < Q; i++) {
        // Reads in a uncoalesced fashion - smaller cost when reading from shared memory
        uint32_t idx = threadIdx.x*Q + i;
        uint32_t elem = shrmem[idx];
        regElem[i] = elem;
    }
}


// From assignment 3-4 - gpu-coalecing
template <class ElTp, int T>
__global__ void
coalsTransposeKer(ElTp* A, ElTp* B, int heightA, int widthA, int colsA) {
  __shared__ ElTp tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

//   printf("x=%d, y=%d\n", x, y);

  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[y*colsA + x];

  __syncthreads();

  x = blockIdx.y * T + threadIdx.x;
  y = blockIdx.x * T + threadIdx.y;

  if( x < heightA && y < widthA )
      B[y*heightA + x] = tile[threadIdx.x][threadIdx.y];
}

template<int Q>
__device__ inline void
scanRegStoreRed(uint32_t* regElem,
                volatile uint32_t* shrmem)
{
    #pragma unroll
    for (int i = 1; i < Q; i++) {
        regElem[i] = regElem[i-1] + regElem[i];
    }
    shrmem[threadIdx.x] = regElem[Q-1];
}

template<class OP>
__device__ void
scan1Block( volatile typename OP::RedElTp* shmem_red) {
    typename OP::RedElTp elm = scanIncBlock<OP>(shmem_red, threadIdx.x);
    __syncthreads();
    shmem_red[threadIdx.x] = elm;
}

template<int B, int Q>
__device__ inline void
mapPrefix2Reg(uint32_t* regElem,
              volatile uint32_t* redShr,
              uint32_t addExtra)
{
    if (threadIdx.x>0) {
        uint32_t prefix = redShr[threadIdx.x-1];
        #pragma unroll
        for (int i = 0; i < Q; i++) {
            regElem[i] += prefix + addExtra;
        }
    }
}

template<int B, int Q, int H, int lgH>
__global__ void
histogramKernel(uint32_t *arr,
                uint32_t *glbHist,
                size_t N,
                int bits_iter,
                int bits)
{
    // __shared__ uint32_t histShr[H];
    // Only bits long!
    extern __shared__ uint32_t histShr[];
    // Initialize shared memory to zero
    // Needs to handle when H<B, so we don't go out of bounds in histShr!
    for (uint32_t i = 0; threadIdx.x + B * i < (1<<bits); i++) {
        histShr[i*B + threadIdx.x] = 0;

    }
    __syncthreads();

    uint32_t block_offset =  blockIdx.x * B*Q;
    uint32_t key_idx;
    for (int q = 0; q < Q; q++) {
        uint32_t arr_idx = block_offset + q * B + threadIdx.x;
        if (arr_idx<N) {
            key_idx = GET_BITS(arr[arr_idx], bits, bits_iter, lgH); // I THINK THIS ONE MESSES UP THE HISTOGRAM IN THE LAST ITERATION IF sizeof(elem) % lgH != 0
            atomicAdd(&histShr[key_idx], 1);
        }
    }
    __syncthreads();
    copyFromShr2Glb<H, B>(blockIdx.x*H, gridDim.x*H, 1<<bits, glbHist, histShr);
}



/**
 * d_ind:       Input array
 * N:           Size of input array
 * hist:        Histogram of input array (hist[B][H])
 * hist_scan:   Scanned histogram (hist_scan[B][H])
 * d_out:       Output array - i_cpu sorted bits
 * i:           Bits iteration (fst lgH bits | snd lgH bits | ...)
**/
template<int B, int Q, int lgH>
__global__ void
partitionScatterKer(uint32_t *d_ind,
                    uint32_t N,
                    uint32_t *hist,
                    uint32_t *hist_scan,
                    uint32_t *d_out,
                    int i,
                    int num_bits)
{
    extern __shared__ uint32_t shrMem[];
    uint32_t *elemShr = shrMem;
    const int H = 1<<num_bits;
    uint32_t *hist_scan_shr = &elemShr[B*Q];
    uint32_t *hist_orig = &hist_scan_shr[H];

    uint32_t regElem[Q];
    uint32_t isT[Q];

    uint32_t loc_ind;
    uint32_t glb_idx;

    //Ideally copy straight to register and figure out way to keep updating local registers instead of copyFromShr2Reg
    copyFromGlb2Shr<B, Q>(blockIdx.x*B*Q, N, d_ind, elemShr);
    __syncthreads();

    int block_offset = blockIdx.x * B*Q;

    for (int k = 0; k < num_bits; k++) {
        // last iteration we need to copy elements into shared memory differently, such that
        // It cause you should read from registers to global memory in the end after this loop has finished and not from shared memory
        // to global memory - also this allows to use less shared memory, since the actual elements will be held in registers we can therefore 
        // overwrite the shared memory with the histogram for instance.
        copyFromShr2Reg<Q>(regElem, elemShr);
        __syncthreads();

        uint16_t acc = 0;
        for (int q = 0; q < Q && block_offset+threadIdx.x*Q+q < N; q++) {
            uint16_t zeroone = (uint16_t)GET_KTH_BIT_UNSET(regElem[q], i, lgH, k);
            acc += zeroone;
            isT[q] = acc;
        }

        elemShr[threadIdx.x] = acc;
        __syncthreads();

        uint16_t res = (uint16_t)scanIncBlock<Add<uint32_t>>(elemShr, threadIdx.x);
        __syncthreads();
        elemShr[threadIdx.x] = res;
        __syncthreads();

        uint32_t max_tid = (N >= block_offset+B*Q) ? B : (N - block_offset + Q - 1) / Q;

        uint16_t split = elemShr[max_tid-1];
        if (threadIdx.x==0) acc=0;
        else acc = elemShr[threadIdx.x-1];

        __syncthreads();

        // Copy back into shared memory
        //  && threadIdx.x*Q+q
        for (int q = 0; q < Q && block_offset+threadIdx.x*Q+q < N; q++) {
            uint16_t zeroone = (uint16_t)GET_KTH_BIT_UNSET(regElem[q], i, lgH, k);
            int pos;
            if (zeroone) {
                pos = isT[q] + acc - 1;
            } else {
                pos = split + (threadIdx.x*Q+q - (isT[q] + acc));
            }
            elemShr[pos] = regElem[q];
        }

        __syncthreads();

    }

    // Maybe move to a copyFromGlb2ShrHis<>() device kernel potentially
    #pragma unroll
    for (int q = 0; q * B + threadIdx.x < H; q++) {
        uint32_t loc_ind = q * B + threadIdx.x;
        uint32_t glb_ind = blockIdx.x*H + loc_ind;
        if (glb_ind < gridDim.x*H) {
            uint32_t elem1 = const_cast<const uint32_t&>(hist[glb_ind]);
            uint32_t elem2 = const_cast<const uint32_t&>(hist_scan[glb_ind]);
            hist_orig[loc_ind] = elem1;
            hist_scan_shr[loc_ind] = elem2;
        }
    }
    
    __syncthreads();
    if (B>=H) {
        scan1Block<Add<uint32_t>>(hist_orig);
    } else {
        int scan_blocks = (H+B-1) / B;
        uint32_t split = 0;
        uint32_t res;
        uint32_t *tmp_ptr = hist_orig;

        for (int i = 0; i < scan_blocks; i++) {
            if (i*B + threadIdx.x < H) res = scanIncBlock<Add<uint32_t>>(tmp_ptr, threadIdx.x);
            __syncthreads();
            if (i*B + threadIdx.x < H) tmp_ptr[threadIdx.x] = res;
            __syncthreads();
            uint32_t max_tid = (H >= (i+1)*B) ? B : (H - i*B);
            split = tmp_ptr[max_tid-1];
            __syncthreads();
            if (i*B + threadIdx.x < H && i>0) tmp_ptr[threadIdx.x] += split;
            __syncthreads();
            tmp_ptr = tmp_ptr + B;
            __syncthreads();
        }
    }


    // Should keep data in registers and write from registers to global memory
    for (int g = 0; g < Q; g++) {
        loc_ind = g * B + threadIdx.x;
        if (block_offset + loc_ind<N) {
            uint32_t bin = GET_BITS(elemShr[loc_ind], num_bits, i, lgH);
            glb_idx = hist_scan_shr[bin] + loc_ind - hist_orig[bin];
            uint32_t elm = const_cast<const uint32_t&>(elemShr[loc_ind]);
            d_out[glb_idx] = elm;
        }
    }
}


#endif
