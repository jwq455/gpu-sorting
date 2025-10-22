#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define GET_BITS(a, mask, shift) ((1<<mask)-1) & (a>>(shift*mask))

int main() {
	srand(time(NULL));
	uint32_t t = (uint32_t)rand();
	printf("t=%u\tt(shifted): %u\n",t, t>>(8*3));

	int x1 = GET_BITS(t, 8, 0);
	int x2 = GET_BITS(t, 8, 1);
	int x3 = GET_BITS(t, 8, 2);
	int x4 = GET_BITS(t, 8, 3);

	printf("x1: %d - x2: %d\n", x1, x2);
	printf("x3: %d - x4: %d\n", x3, x4);

	uint32_t t2 = 4294967295;
	uint32_t t_shift = t2>>(8*3);
	uint32_t mask = (1<<8)-1;

	printf("t: %u - t_shift: %u - mask: %u - GET_BITS()=%u\n", t2, t_shift, mask, GET_BITS(t2, 8, 3));

	return 0;
}
