#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define GET_BITS(a, mask, shift) ((1<<mask)-1) & (a>>(shift*mask))
#define GET_KTH_BIT(a, i, lgH, k) (1) & (a>>(i*lgH+k)) 
#define GET_KTH_BIT_UNSET(a, i, lgH, k) (1) ^ ((1) & (a>>(i*lgH+k))) 

template<int BLOCKS, int Q, int B>
void histogram(uint32_t *hist, uint32_t *arr_inp, const int H, const int lgH, const int N, int i) {
    int key_idx;
    for (int b = 0; b < BLOCKS; b++) {
        for (int bb = 0; bb < B; bb++) {
            for (int g = 0; g < Q; g++) {
                key_idx = GET_BITS(arr_inp[b*B*Q+bb*Q+g], lgH, i);
                hist[b*H+key_idx]++;
            }
        }
    }
}

void transpose(uint32_t *hist_tr, uint32_t *hist, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            hist_tr[j*rows+i] = hist[i*cols+j];
        }
    }
}

void scanHist(uint32_t *hist_scan, uint32_t *hist_tr, const int rows, const int cols) {
    uint32_t elem;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (i==0 && j==0) {
                hist_scan[i*cols+j] = hist_tr[i*cols+j];
                continue;
            }        
            hist_scan[i*cols+j] = hist_scan[(i*cols+j)-1] + hist_tr[i*cols+j];
        }
    }
}

template<int B, int Q>
void copyFromGlb2Shr(const uint32_t glb_offset,
                    const uint32_t N,
                    uint32_t* d_ind,
                    uint32_t* shrmem)
{
    for (int bb = 0; bb < B; bb++) {
        for (int i = 0; i < Q; i++) {
            uint32_t loc_ind = i * B + bb;
            uint32_t glb_ind = glb_offset + loc_ind;
            if (glb_ind<N) {
                shrmem[loc_ind] = d_ind[glb_ind];
            }
        }
    }
}

// template<int B, int Q, int lgH>
// void copyFromShr2Reg(uint32_t regElem[B][3][Q], 
//                      uint32_t *shrmem,
//                      int i_bits,
//                      int k_bit)
// {
//     for (int bb = 0; bb < B; bb++) {
//         for (int i = 0; i < Q; i++) {
//             // Reads in a uncoalesced fashion - smaller cost when reading from shared memory
//             uint32_t idx = bb*Q + i;
//             uint32_t elem = shrmem[idx];
//             regElem[bb][0][i] = elem;
//             regElem[bb][1][i] = 1^GET_KTH_BIT(elem, i_bits, lgH, k_bit);
//             regElem[bb][2][i] = GET_KTH_BIT(elem, i_bits, lgH, k_bit);
//         }
//     }
// }

template<int B, int Q>
void copyFromShr2Reg(uint32_t *regElem, 
                     uint32_t *shrmem)
{
    for (int bb = 0; bb < B; bb++) {
        for (int i = 0; i < Q; i++) {
            // Reads in a uncoalesced fashion - smaller cost when reading from shared memory
            uint32_t idx = bb*Q + i;
            uint32_t elem = shrmem[idx];
            regElem[bb*Q+i] = elem;
            // regElem[1][i] = 1^GET_KTH_BIT(elem, i_bits, lgH, k_bit);
            // regElem[2][i] = GET_KTH_BIT(elem, i_bits, lgH, k_bit);
        }
    }
}

template<int B, int Q>
void scanRegStoreRed(uint32_t regElem[B][3][Q],
                     uint32_t* redShr)
{
    for (int bb = 0; bb < B; bb++) {
        for (int i = 1; i < Q; i++) {
            regElem[bb][1][i] = regElem[bb][1][i-1] + regElem[bb][1][i];
        }
        redShr[bb] = regElem[bb][1][Q-1];
    }
}

int main() {
    // Input/output array
    const int Q = 8;
    const int B = 32;
    const int N = (Q*B)*2; // 512 -> 2 blocks
    const int blocks = (N + Q*B - 1) / (Q*B);
    printf("blocks: %d\n", blocks);
    uint32_t *d_A = (uint32_t *)malloc(sizeof(uint32_t)*N); 
    uint32_t *d_B = (uint32_t *)calloc(N, sizeof(uint32_t)); 

    for (uint32_t i = 0; i < N; i++) {
        d_A[i] = ((N-1)-i) % 256;
    }

    printf("Input array\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", d_A[i]);
    }
    printf("\n");

    // Historgram
    const int lgH = 4;
    const int H = 1<<lgH; // 256
    uint32_t *hist = (uint32_t *)calloc(2*H, sizeof(uint32_t)); 
    uint32_t *hist_tr = (uint32_t *)calloc(2*H, sizeof(uint32_t)); 
    uint32_t *hist_scan = (uint32_t *)calloc(2*H, sizeof(uint32_t));

    histogram<blocks, Q, B>(hist, d_A, H, lgH, N, 0);
    printf("Histogram\n");
    for (int i = 0; i < blocks; i++) {
        printf("Block: %d\n", i);
        for (int j = 0; j < H; j++) {
            printf("%d ", hist[i*H+j]);
        }
        printf("\n\n");
    }

    transpose(hist_tr, hist, blocks, H);
    printf("Histogram transposed\n");
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < blocks; j++) {
            printf("%d ", hist_tr[i*blocks+j]);
        }
        printf("\n");
    }

    scanHist(hist_scan, hist_tr, H, blocks);
    printf("Histogram scanned\n");
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < blocks; j++) {
            printf("%d ", hist_scan[i*blocks+j]);
        }
        printf("\n");
    }

    transpose(hist_tr, hist_scan, H, blocks);
    printf("Scanned histogram transposed\n");
    for (int i = 0; i < blocks; i++) {
        for (int j = 0; j < H; j++) {
            printf("%d ", hist_tr[i*H+j]);
        }
        printf("\n");
    }
    printf("\n");

    uint32_t *elemShr1 = (uint32_t *)calloc(B*Q, sizeof(uint32_t));
    copyFromGlb2Shr<B, Q>(0*B*Q, N, d_A, elemShr1);
    printf("Shared Memory 1\n");
    for (int i = 0; i < B*Q; i++) {
        printf("%d ", elemShr1[i]);
    }
    printf("\n\n");

    uint32_t *elemShr2 = (uint32_t *)calloc(B*Q, sizeof(uint32_t));
    copyFromGlb2Shr<B, Q>(1*B*Q, N, d_A, elemShr2);
    printf("Shared Memory 2\n");
    for (int i = 0; i < B*Q; i++) {
        printf("%d ", elemShr2[i]);
    }
    printf("\n\n");

    uint32_t *regElem = (uint32_t *)calloc(Q*B, sizeof(uint32_t));
    uint32_t *isT = (uint32_t *)calloc(Q*B, sizeof(uint32_t));
    int k_bit = 2;

    copyFromShr2Reg<B, Q>(regElem, elemShr1);
    printf("Register memory\n");
    for (int i = 0; i < 4; i++) {
        printf("Thrad %d\n", i);
        for (int j = 0; j < Q; j++) {
            printf("Elem: %d\n", regElem[i*Q+j]);
        }
    }
    printf("\n\n");

    uint16_t acc[B];

    for (int bb = 0; bb < B; bb++) {
        acc[bb] = 0;
        if (bb<5) printf("Thread_%d: isT\n", bb);
        for (int q = 0; q < Q; q++) {
            uint16_t zeroone = (uint16_t)GET_KTH_BIT_UNSET(regElem[bb*Q+q], 0, lgH, k_bit);
            acc[bb] += zeroone;
            isT[bb*Q+q] = acc[bb];
            if (bb<5) printf("Bit: %d\tisT[%d]=%d\n", zeroone, q, acc[bb]);
        }
        if (bb<5) printf("\n");
    }

    for (int bb = 0; bb < B; bb++) {
        elemShr1[bb] = acc[bb];
    }

    // SCAN shared memory
    for (int bb = 1; bb < B; bb++) {
        elemShr1[bb] += elemShr1[bb-1];
    }

    uint16_t split = elemShr1[B-1];
    acc[0]=0;
    for (int bb = 1; bb < B; bb++) acc[bb] = elemShr1[bb-1];

    printf("split=%d\n", split);
    for (int bb = 0; bb < 4; bb++) {
        printf("acc[%d] = %d\n", bb, acc[bb]);
    }

    for (int bb = 0; bb < 4; bb++) {
        for (int q = 0; q < Q; q++) {
            uint16_t zeroone = (uint16_t)GET_KTH_BIT_UNSET(regElem[q], 0, lgH, k_bit);
            // acc[bb] += zeroone;
            int pos;
            if (zeroone) {
                pos = isT[bb*Q+q] + acc[bb] - 1;
            } else {
                pos = split + (bb*Q+q - (isT[bb*Q+q] + acc[bb]));
            }
            printf("POS: %d\tRegElm[%d]=%d\n", pos, q, regElem[bb*Q+q]);
            elemShr1[pos] = regElem[bb*Q+q];
        }
    }

    // for (int k = 0; k < lgH; k++) {
    // // Copy Q elements from shared to register memory
    // // !!!Might need to make sure not to read from shared memory not written form global memory!!!
    // // We really need the k'th bit with the predicate x==0 applied to and later its opposite
    // // Perhaps we could during the copying just pick out these bit value
    // // What about when it comes to writing the value back into the shared memory? Do we need to have
    // // have the original value in registers?? probably
    // copyFromShr2Reg<B, Q>(regElem, elemShr1);

    // // uint32_t idx = threadIdx.x*Q + i;
    // //     uint32_t elem = shrmem[idx];
    // for (int bb = 0; bb < B; bb++) {
    //     uint16_t acc = 0;
    //     for (int q = 0; q < Q; q++) {
    //         uint16_t zeroone = (uint16_t)GET_KTH_BIT_UNSET(regElem[bb*Q+q], 0, lgH, k);
    //         acc += zeroone;
    //         isT[bb*Q+q] = acc;
    //     }
    // }

    // elemShr[threadIdx.x] = acc;
    // __syncthreads();

    // uint16_t res = scanIncBlock<Add<uint16_t>>(elemShr, threadIdx.x);
    // __syncthreads();
    // elemShr[threadIdx.x] = res;
    // __syncthreads();

    // uint16_t split = elemShr[B-1];
    // if (threadIdx.x==0) acc=0;
    // else acc = elemShr[threadIdx.x-1];

    // // uint32_t split = elemShr[B-1];

    // // Copy back into shared memory
    // for (int q = 0; q < Q; q++) {
    //     uint16_t zeroone = (uint16_t)GET_KTH_BIT_UNSET(regElem[q], i, lgH, k);
    //     acc += zeroone;
    //     int pos;
    //     if (zeroone) {
    //         pos = isT[q] + acc - 1;
    //     } else {
    //         pos = split + (threadIdx.x*Q+q - (isT[q] + acc));
    //     }
    //     elemShr[pos] = regElem[q];
    // }
    
    // copy back into registers

    // }
    
    return 0;
}