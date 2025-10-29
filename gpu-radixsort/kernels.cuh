#ifndef RADIX_KERS
#define RADIX_KERS

#include <cuda_runtime.h>
#include "pbb_kernels.cuh"
#include "../helper.h"

#define GET_BITS(a, mask, shift) ((1<<mask)-1) & (a>>(shift*mask))
// Gets the i*lgH+k'th bit of 'a'
#define GET_KTH_BIT_UNSET(a, i, lgH, k) (1) ^ ((1) & (a>>(i*lgH+k))) 


// Rewrite such that you don't need CHUNK
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
coalsTransposeKer(ElTp* A, ElTp* B, int heightA, int widthA) {
  __shared__ ElTp tile[T][T+1];

  int x = blockIdx.x * T + threadIdx.x;
  int y = blockIdx.y * T + threadIdx.y;

  if( x < widthA && y < heightA )
      tile[threadIdx.y][threadIdx.x] = A[y*widthA + x];

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

// template<int B>
// __device__ inline void
// scanRedShr(volatile uint32_t* redShr,
//            uint32_t idx)
// {
//     int offset = 1;
//     for (int d = B >> 1; d > 0; d >> = 1)
//     // build sum in place up the tree
//     {
//         __syncthreads();
//         if (idx < d) {         
//             int ai = offset * (2 * idx + 1) - 1;
//             int bi = offset * (2 * idx + 2) - 1;
            
//             redShr[bi] += redShr[ai];
//         }
//         offset *= 2;
//     }
//     if (idx == 0)
//     {
//       redShr[B - 1] = 0;
//     } // clear the last element
//     for (int d = 1; d < B; d *= 2) // traverse down tree & build scan
//     {
//         offset >> = 1;
//         __syncthreads();
//         if (idx < d) {
//             int ai = offset * (2 * idx + 1) - 1;
//             int bi = offset * (2 * idx + 2) - 1;
//             float t = redShr[ai];
//             redShr[ai] = redShr[bi];
//             redShr[bi] += t;
//         }
//     }
//     __syncthreads();
// }


template<class OP>
__device__ void
scan1Block( volatile typename OP::RedElTp* shmem_red) {
    typename OP::RedElTp elm = scanIncBlock<OP>(shmem_red, threadIdx.x);
    __syncthreads(); // Not sure about this sync, I am afraid if we don't sync we will mess up the scanIncBlock() call
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
    // Not sure this sync has to be here
    __syncthreads();

    uint32_t block_offset =  blockIdx.x * B*Q;
    uint32_t key_idx;
    for (int q = 0; q < Q; q++) {
        uint32_t arr_idx = block_offset + q * B + threadIdx.x;
        if (arr_idx<N) {
            key_idx = GET_BITS(arr[arr_idx], lgH, bits_iter);
            atomicAdd(&histShr[key_idx], 1);
        }
    }
    __syncthreads();
    copyFromShr2Glb<H, CHUNK>(blockIdx.x*H, gridDim.x*H, glbHist, histShr);
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
                    int i)
{
    __shared__ uint32_t elemShr[B*Q];
    uint32_t regElem[Q];
    uint32_t isT[Q];

    uint32_t loc_ind;
    uint32_t glb_idx;

    // Perhaps copy straight to register and figure out way to keep updating local registers instead of copyFromShr2Reg
    copyFromGlb2Shr<B, Q>(blockIdx.x*B*Q, N, d_ind, elemShr);
    __syncthreads();

    int block_offset = blockIdx.x * B*Q;

    for (int k = 0; k < lgH; k++) {
        // last iteration we need to copy elements into shared memory differently, such that -! I DON'T UNDERSTAND THIS NOT SURE !-
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

        // if (threadIdx.x==0 && k==0) printf("maxx_tid=%d\n", max_tid);

        uint16_t split = elemShr[max_tid-1];
        if (threadIdx.x==0) acc=0;
        else acc = elemShr[threadIdx.x-1];

        __syncthreads();

        // Copy back into shared memory
        //  && threadIdx.x*Q+q
        for (int q = 0; q < Q && block_offset+threadIdx.x*Q+q < N; q++) {
            uint16_t zeroone = (uint16_t)GET_KTH_BIT_UNSET(regElem[q], i, lgH, k);
            // acc += zeroone;
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


    const int H = 1<<lgH;
    const int chunk = (H+B-1) / B;

    // You only need two the scanned histogram and one for the original histogram
    /*
    (histo_scan[blockixd][bin] - histo_orig[blockixd][bin]) + (g*B+threadIdx.x - histo_orig_exc_scan[blockixd][bin])

    = histo_scan[blockixd][bin] + g*B+threadIdx.x - histo_orig[blockixd][bin] - histo_orig_exc_scan[blockixd][bin]

    //
        histo_orig[blockixd][bin] + histo_orig_exc_scan[blockixd][bin] = histo_orig_inc_scan[blockixd][bin]
    //

    = histo_scan[blockixd][bin] + g*B+threadIdx.x - histo_orig_inc_scan[blockixd][bin]
    
    */

    __shared__ uint32_t hist_orig[H];
    __shared__ uint32_t hist_orig_scan[H];
    __shared__ uint32_t hist_scan_shr[H];


    // Maybe move to a copyFromGlb2ShrHis<>() device kernel

    // copyFromGlb2Shr<H, chunk>(blockIdx.x*H, gridDim.x*H, hist, hist_orig);
    // copyFromGlb2Shr<H, chunk>(blockIdx.x*H, gridDim.x*H, hist, hist_orig_scan);
    // copyFromGlb2Shr<H, chunk>(blockIdx.x*H, gridDim.x*H, hist_scan, hist_scan_shr);
    #pragma unroll
    for (int q = 0; q < chunk; q++) {
        uint32_t loc_ind = q * B + threadIdx.x;
        uint32_t glb_ind = blockIdx.x*H + loc_ind;
        if (loc_ind<H && glb_ind < gridDim.x*H) {
            uint32_t elem1 = const_cast<const uint32_t&>(hist[glb_ind]);
            uint32_t elem2 = const_cast<const uint32_t&>(hist_scan[glb_ind]);
            hist_orig[loc_ind] = elem1;
            hist_orig_scan[loc_ind] = elem1;
            hist_scan_shr[loc_ind] = elem2;
        }
    }

    __syncthreads();
    scan1Block<Add<uint32_t>>(hist_orig_scan);
    __syncthreads();

    for (int g = 0; g < Q; g++) {
        loc_ind = g * B + threadIdx.x;
        if (block_offset + loc_ind<N) {
            uint32_t bin = GET_BITS(elemShr[loc_ind], lgH, i);
            glb_idx = (hist_scan_shr[bin] - hist_orig[bin]) + (loc_ind - ((bin==0) ? 0 : hist_orig_scan[bin-1]));
            uint32_t elm = const_cast<const uint32_t&>(elemShr[loc_ind]);
            d_out[glb_idx] = elm;
        }
    }
}


#endif
