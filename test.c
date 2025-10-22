#include <stdio.h>
#include <stdlib.h>

#define GET_BITS(a, mask, shift) ((1<<mask)-1) & (a>>(shift*mask))


int main() {
	int x1 = GET_BITS(256, 8, 0);
	int x2 = GET_BITS(256, 8, 1);
	int x3 = GET_BITS(0xff0000, 8, 2);
	int x4 = GET_BITS(0xff000000, 8, 3);

	printf("x1: %d - x2: %d\n", x1, x2);
	printf("x3: %d - x4: %d\n", x3, x4);

	return 0;
}
