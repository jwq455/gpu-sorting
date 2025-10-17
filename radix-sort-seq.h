#ifndef RADIX_SEQ
#define RADIX_SEQ

#include "math.h"

int digit(int a, int base, int n) {
    return a / (int)pow(base, n - 1) % base;
}

/*
 * Counting Sort - Sorting array `A` on digit `i` assuming digit `i` has base (radix) `k`
 *
 * args:
 *      A: Array of type T
 *      B: Array A sorted of size sizeA
 *      sizeA: Length of array A
 *      k: Possible values of each element in A
 *      ith: Sort on the i'th digit of each element in A ith>1
 * 
 * Returns: 
 *      Array B, that is array A sorted
 * Note:
 *      Type `T` must be a positive integer or floating point value 
*/
template<class T>
void countingSort(T* A, T* B, int sizeA, int k, int ith) {
    T* C = (T*)malloc(sizeof(T)*k);
    // Init C = { 0 }
    for (int i = 0; i < k; i++) C[i] = 0;
    // Count how many elements of A are equal to i
    for (int j = 0; j < sizeA; j++) C[digit(A[j], k, ith)]++;
    // Inclusive scan of C - Number of elements less than or equal to i
    for (int i = 1; i < k; i++) C[i] = C[i]+C[i-1];
    for (int i = 0; i < k; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");
    // Inserts A elements into B in a sorted stable order
    for (int j = sizeA-1; j>=0; j--) {
        int idx = digit(A[j], k, ith);
        // printf("Idx: %d\n", idx);
        printf("A[%d]=%d\n", j, A[j]);
        printf("C[%d]=%d\n", idx, C[idx]);
        B[C[idx]-1] = A[j];
        C[idx]--;
    }
    printf("\n");
}

template<class T>
void radixSortSeq(T* A, T* B, int sizeA, int d) {
    T *tmp;
    for (int i = 1; i <=d; i++) {
        countingSort(A, B, sizeA, 10, i);
        tmp = A;
        A = B;
        B = tmp;
    }
}

#endif