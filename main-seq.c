using namespace std;

#include "helper.h"
#include "radix-sort-seq.h"

template<class T>
int number_of_digits(T* A, uint32_t size_A) {
    int max_digit = 10;
    int zeros;
    while (max_digit) {
        for (int i = 0; i < size_A; i++) {
            zeros = (int)pow(10, max_digit-1);
            if ((A[i] / zeros) != 0) return max_digit;
        }
        max_digit--;
    }
    return 0;
}

template<class T>
void runSeq(uint32_t size_A) {
    srand(2025);

    unsigned long long mem_size_A = sizeof(T) * size_A;
    T* A = (T*) malloc(mem_size_A);
    T* B = (T*) malloc(mem_size_A);

    randomInit<T>(A, size_A);
    zeroInit<T>(B, size_A);

    int d = number_of_digits<T>(A, size_A);
    if (d==0) {
        printf("Maximmum number of digits equals zero!?\n");
        exit(1);
    }
    
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL); 
    
    radixSortSeq<T>(A, B, size_A, d);

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec);
    double flopsPerArray = 4.0 * size_A * d;
    double gigaFlops = (flopsPerArray * 1.0e-3f) / elapsed; 
    
    printf("Sequential version of radix sort runs in: %.1f microsecs, GFlops/sec: %.2f\n", elapsed, gigaFlops);

    for (int i = 0; i < size_A; i++) {
        printf("%d ", A[i]);
    }
    printf("\n");

    for (int i = 0; i < size_A; i++) {
        printf("%d ", B[i]);
    }
    printf("\n");

    validate<T>(B, size_A);

}

int main(int argc, char *argv[]) {
    if (argc!=2) {
        printf("Usage: %s length-A\n", argv[0]);
        exit(1);
    }

    int size_A = atoi(argv[1]);

    printf("Running CPU sequential Versions of radix sort\n");

    runSeq<uint32_t>(size_A);

    return 0;
    
}