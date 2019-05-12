#include <stdio.h>

#ifndef DUEL
#define DUEL

__device__ int duel(char* T, char* P, int* W, int i, int j, int wsize, int psize, int tsize);

#endif

#ifndef DUEL_DEF
#define DUEL_DEF

//Duel Function ***************************************************************
__device__ int duel(char* T, char* P, int* W, int i, int j, int wsize, int psize, int tsize){
	if (j-i >= wsize){printf("K ERROR\n");}
	int k = W[j-i];
	if (j+k >= tsize){
		return i;
	}
	if (T[j+k] != P[k]){ 
		return i;
	} else {
		return j;
	}
}

#endif
