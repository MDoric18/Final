#include <stdio.h>

#ifndef WIT
#define WIT

__global__ void witnessGPU(char*, int, int*, int);
__host__ void witnessCPU(char*, int, int*, int);

#endif

#ifndef WIT_DEF
#define WIT_DEF

//Witness Array GPU *********************************************************
__global__ void witnessGPU(char* P, int psize, int* W, int wsize){
	int ID = threadIdx.x + blockIdx.x*blockDim.x;
	int i = ID / wsize;
	int k = ID % wsize;
	if ((i+k < psize)&&(i < wsize)){
		if (P[k] != P[i+k]){
			W[i] = k; 		
		}
	}
	W[0] = 0; 
}

__host__ void witnessCPU(char* P, int psize, int* W, int wsize){
	W[0] = 0; 
	for (int i=1; i<wsize; i++){ 
		for (int k=1; i+k<psize; k++){
			if (P[k] != P[i+k]){
				W[i] = k; 
				break; 
			}
		}
	}
}

#endif
