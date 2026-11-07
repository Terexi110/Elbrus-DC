#include <stdio.h>

int main(){
	int a, b, c;
	scanf("%d", a);
	scanf("%d", b);
	if (a > b) {
		c = a;
	} else if (a < b) {
		c = b;
	} else {
		c = a + b;
	}
	printf("%d", c);
	return 0;
}
