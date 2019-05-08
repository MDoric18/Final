#include <stdio.h>
#include "duel.cuh"

#ifndef MULT
#define MULT

__global__ void createTreeLevel(int*, int, char*, int, char*, int, int*, int);
__global__ void search_Finish(int, char*, int, char*, int, int*, int*); 

#endif

#ifndef MULT_DEF
#define MULT_DEF

//Multiple kernel parallel search ***********************************************************
__global__ void createTreeLevel(int* W, int wsize, char*P, int psize, char* T, int tsize, int* tree, int s){	
	//Setup
	int ID = threadIdx.x + blockIdx.x*blockDim.x;
	int tID = ID % wsize;
	int bID = ID / wsize;
	int blockStart = bID*wsize;

	//If this is the first level, add initial leaves
	if ((s == 0)&&(ID < tsize)){
		tree[ID] = ID; 
	}

	//Create current level
	if (s > 0){
		if ((2*s*tID + s < wsize)&&(2*s*tID + s + blockStart < tsize)){
			int i = tree[2*s*tID + blockStart];
			int j = tree[2*s*tID + s + blockStart];
			tree[2*s*tID + blockStart] = duel(T, P, W, i, j);
			if (2*s*tID + s + blockStart >= tsize){printf("OoB ERROR\n");}
		}
	}
}

__global__ void search_Finish(int wsize, char* P, int psize, char* T, int tsize, int* match, int* tree){
	//Setup
	int ID = threadIdx.x + blockIdx.x*blockDim.x;

	//Check for pattern
	if (ID*wsize < tsize){
		int i = 0;
		int m = tree[ID*wsize]; 
		while (i < psize){
			if (i + m >= tsize){
				m = -1; //Out of bounds
				break;
			}		
			if (T[i+m] != P[i]){
				m = -1;
				break;
			}
			i++; 
		}

		if (ID >= tsize){printf("error in match\n");}
		//Store result
		match[ID] = m;  
	}
}

#endif
