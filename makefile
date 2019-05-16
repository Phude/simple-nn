CC = clang
CFLAGS = -Wall
LDFLAGS = -lm

main:
	$(CC) src/mlp.c $(CFLAGS) $(LDFLAGS)
