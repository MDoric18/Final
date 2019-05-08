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
	//if (ID == 0){printf("In create tree\n"); }
	int tID = ID % wsize;
	int bID = ID / wsize;
	int blockStart = bID*wsize;

	//If this is the first level, add initial leaves
	if (s == 0){
		tree[ID] = ID; 
	}

	//Create current level
	if (s > 0){
		if ((2*s*tID + s < wsize)&&(2*s*tID + s + blockStart < tsize)){
			int i = tree[2*s*tID + blockStart];
			int j = tree[2*s*tID + s + blockStart];
			tree[2*s*tID + blockStart] = duel(T, P, W, i, j);
		}
	}
	//if (ID == 0){printf("Created tree\n"); }
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

		//Store result
		match[ID] = m; 
	}
	//if (ID == 0){printf("Finished search w/psize = %d\n", psize); }
	/*//DEBUG
	__syncthreads(); 
	if (ID == 0){
		for (int j=0; j<tsize/wsize; j++){
			printf("%d ", match[j]);
		}
		printf("\n"); 
	}//*/
}

#endif