#include <stdio.h>
#include "duel.cuh"

#ifndef MULT
#define MULT

__global__ void setup_Tree(int* tree, int tsize); 
__global__ void createTreeLevel(int*, int, char*, int, char*, int, int*, int);
__global__ void search_Finish(int, char*, int, char*, int, int*, int*); 

#endif

#ifndef MULT_DEF
#define MULT_DEF

//Multiple kernel parallel search 
//***********************************************************
__global__ void setup_Tree(int* tree, int tsize){
	int ID = threadIdx.x + blockIdx.x*blockDim.x;
	if (ID < tsize){
		tree[ID] = ID; 
	}
}

__constant__ char TC[2048*32]; //Max constant memory allowed

__host__ void setConstantMem(char* text, int tsize){
	cudaMemcpyToSymbol(TC, text, tsize*sizeof(char));  
}


//Global Memory Version
//***************************************************************
__global__ void createTreeLevel(int* W, int wsize, char*P, int psize, char* T, int tsize, int* tree, int s){	
	//Setup
	int ID = threadIdx.x + blockIdx.x*blockDim.x;
	int tID = ID % wsize;
	int bID = ID / wsize;
	int blockStart = bID*wsize;
	int cap = tsize - blockStart; 

	//Create current level
	int ind = 2*s*tID + blockStart;
	if ((2*s*tID + s < wsize)&&(cap > wsize)){
		if (ind + s >= tsize){printf("OoB ERROR\n");}
		int i = tree[ind];
		int j = tree[ind + s];
		int temp = duel(T, P, W, i, j, wsize, psize, tsize);
		tree[ind] = temp; 
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

		//Store result
		match[ID] = m;
	}
}


//Constant Memory version
//******************************************************************
__global__ void createTreeLevel_Constant(int* W, int wsize, char*P, int psize, int tsize, int* tree, int s){	
	//Setup
	int ID = threadIdx.x + blockIdx.x*blockDim.x;
	int tID = ID % wsize;
	int bID = ID / wsize;
	int blockStart = bID*wsize;
	int cap = tsize - blockStart; 

	//Create current level
	int ind = 2*s*tID + blockStart;
	if ((2*s*tID + s < wsize)&&(cap > wsize)){
		if (ind + s >= tsize){printf("OoB ERROR\n");}
		int i = tree[ind];
		int j = tree[ind + s];
		//Duel Function Equivalent
		int temp = i; 
		int k = W[j-i];
		if (TC[j+k] == P[k]){ 
			temp = j; 
		}
		//End Duel
		tree[ind] = temp; 
	}
}

__global__ void search_Finish_Constant(int wsize, char* P, int psize, int tsize, int* match, int* tree){
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
			if (TC[i+m] != P[i]){
				m = -1;
				break;
			}
			i++;
		} 

		//Store result
		match[ID] = m;
	}
}

#endif
