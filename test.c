#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define GET_BITS(a, mask, shift) ((1<<mask)-1) & (a>>(shift*mask))
#define GET_KTH_BIT(a, i, lgH, k) (1) & (a>>(i*lgH+k)) 

int main() {
	srand(time(NULL));
	uint32_t t = (uint32_t)rand();
	printf("t=%u\tt(shifted): %u\n",t, t>>(8*3));

	int x1 = GET_BITS(t, 8, 0);
	int x2 = GET_BITS(t, 8, 1);
	int x3 = GET_BITS(t, 8, 2);
	int x4 = GET_BITS(t, 8, 3);


	int x5 = GET_KTH_BIT(9, 0, 8, 0); // ==1
	int x6 = GET_KTH_BIT(9, 0, 8, 1); // ==0
	int x7 = GET_KTH_BIT(10, 0, 8, 1); // ==1
	int x8 = GET_KTH_BIT(256, 1, 8, 0); // ==1

	printf("x1: %d - x2: %d\n", x1, x2);
	printf("x3: %d - x4: %d\n", x3, x4);
	printf("x5: %d - x6: %d\n", x5, x6);
	printf("x7: %d - x8: %d\n", x7, x8);

	uint32_t t2 = 4294967295;
	uint32_t t_shift = t2>>(8*3);
	uint32_t mask = (1<<8)-1;

	printf("t: %u - t_shift: %u - mask: %u - GET_BITS()=%u\n", t2, t_shift, mask, GET_BITS(t2, 8, 3));

	return 0;
}
