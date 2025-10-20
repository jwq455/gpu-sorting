/*
 * Helper code stolen from:
 *        "https://github.com/diku-dk/pmph-e2025-pub/tree/main/weeklies/assignment-3-4"
 * */

#ifndef HELPER
#define HELPER

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <time.h>

#if 0
typedef int        int32_t;
typedef long long  int64_t;
#endif

typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define min(a,b) ( ((a)<(b))? (a) : (b) )

#define gpuAssert(code) { __cudassert((code), __FILE__, __LINE__); }
#define gpuCheck(code) { __cudassert((code), __FILE__, __LINE__, false); }


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

#endif