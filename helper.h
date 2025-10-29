/*
 * Helper code stolen from:
 *        "https://github.com/diku-dk/pmph-e2025-pub/tree/main/weeklies/assignment-3-4"
 * */

#ifndef HELPER_H
#define HELPER_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "string.h"
#include <sys/time.h>
#include <time.h>
#include <stdint.h>

#define GPU_RUNS            250

#define TILE_SIZE           32

#define DEBUG_INFO          true

#define lgWARP              5
#define WARP                (1<<lgWARP)

#ifndef ELEMS_PER_THREAD
#define ELEMS_PER_THREAD    24
#endif

#ifndef WARP_REDUCE
#define WARP_REDUCE         1
#endif


#if 0
typedef int        int32_t;
typedef long long  int64_t;
#endif

#define min(a,b) ( ((a)<(b))? (a) : (b) )

#define gpuAssert(code) { __cudassert((code), __FILE__, __LINE__); }
#define gpuCheck(code) { __cudassert((code), __FILE__, __LINE__, false); }

uint32_t MAX_HWDTH;
uint32_t MAX_BLOCK;
uint32_t MAX_SHMEM;

cudaDeviceProp prop;

void initHwd() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    cudaGetDeviceProperties(&prop, 0);
    MAX_HWDTH = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    MAX_BLOCK = prop.maxThreadsPerBlock;
    MAX_SHMEM = prop.sharedMemPerBlock;

    if (DEBUG_INFO) {
        printf("Device name: %s\n", prop.name);
        printf("Number of hardware threads: %d\n", MAX_HWDTH);
        printf("Max block size: %d\n", MAX_BLOCK);
        printf("Shared memory size: %d\n", MAX_SHMEM);
        puts("====");
    }
}


void __cudassert(cudaError_t code,
                 const char *file,
                 int line,
                 bool do_abort_on_err = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPU Error in %s (line %d): %s\n",
            file, line, cudaGetErrorString(code));
    if (do_abort_on_err)
        exit(1);
  }
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

template<class T>
void randomInit(T* data, uint64_t size) {
    for (uint64_t i = 0; i < size; i++)
        data[i] = (T)rand();
}

template<class T>
void zeroInit(T* data, uint64_t size) {
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < size; i++)
        data[i] = 0;
}

// This assumes that type T is comparable with <, > & == 
template<class T>
bool validate(T* A, const uint64_t sizeA){
    for(uint64_t i = 0; i < sizeA-1; i++) {
        if (A[i] > A[i+1]) {
            printf("INVALID RESULT at index %llu: %d vs %d\n", i, A[i], A[i+1]);
            return false;
        }
    }
    printf("VALID RESULT!\n");
    return true;
}

template<class ElTp>
int validateTranspose(ElTp* A, ElTp* trA, const uint32_t rowsA, const uint32_t colsA){
  int valid = 1;
  for(uint64_t i = 0; i < rowsA; i++) {
    for(uint64_t j = 0; j < colsA; j++) {
      if(trA[j*rowsA + i] != A[i*colsA + j]) {
        printf("row: %llu, col: %llu, A: %.4f, trA: %.4f\n"
              , i, j, A[i*colsA + j], trA[j*rowsA + i] );
        valid = 0;
        break;
      }
    }
    if(!valid) break;
  }
  if (valid) printf("GPU TRANSPOSITION   VALID!\n");
  else       printf("GPU TRANSPOSITION INVALID!\n");
  return valid;
}

#endif
