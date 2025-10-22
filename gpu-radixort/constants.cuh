#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <sys/time.h>
#include <time.h>
#include <stdint.h>

#define DEBUG_INFO  true

#define lgWARP      5
#define WARP        (1<<lgWARP)

#ifndef ELEMS_PER_THREAD
#define ELEMS_PER_THREAD   24
#endif

#ifndef WARP_REDUCE
#define WARP_REDUCE         1
#endif

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


#endif // CONSTANTS_H

