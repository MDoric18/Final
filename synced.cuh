#include <stdio.h>
#include "duel.cuh"

#ifndef SYNC
#define SYNC

__global__ void search_synced(int*, int, char*, int, char*, int, int*, int*);
__global__ void search_synced_shared(int*, int, char*, int, char*, int, int*);

#endif

#ifndef SYNC_DEF
#define SYNC_DEF

//Synced Parallel Search *****************************************************************
__global__ void search_synced(int* W, int wsize, char* P, int psize, char* T, int tsize, int* match, int* tree){ 
	
	//We will use the thread indices to correspond to i's and j's to create the results of the duel tree in the given array, tree. Since we don't need to hold onto the results of the tree, we can just have tree be the size of the witness array for each block, and then call duel repeatedly on every two indices until one is left. Similar to what we did with the vector sum. 
 
	int ID = threadIdx.x + blockIdx.x*wsize;
	int blockStart = blockIdx.x*wsize; 
	if (ID < tsize){
		tree[ID] = ID; 
	}

	__syncthreads();

	for (int s = 1; s < wsize; s = 2*s){
		if ((2*s*threadIdx.x+s < wsize)&&(2*s*threadIdx.x+s+blockStart < tsize)){
			int i = tree[2*s*threadIdx.x + blockStart];
			int j = tree[2*s*threadIdx.x + s + blockStart];
			tree[2*s*threadIdx.x + blockStart] = duel(T, P, W, i, j);
			if (2*s*threadIdx.x + s + blockStart >= tsize){printf("OoB ERROR\n");}		
		}
		__syncthreads(); 
	}

	//Now for each block, check the one location where the pattern may possibly occur using brute force. Use thread 0 to finish. 
	if (threadIdx.x == 0){
		int i = 0;
		int m = tree[blockStart]; 
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

		//Store the result
		if (blockIdx.x > tsize){printf("M ERROR SS\n");}
		match[blockIdx.x] = m;
	}
}

//Synced Parallel Search w/ Shared Mem **********************************************************
__global__ void search_synced_shared(int* W, int wsize, char* P, int psize, char* T, int tsize, int* match){ 
	
	//We will use the thread indices to correspond to i's and j's to create the results of the duel tree in the given array, tree. Since we don't need to hold onto the results of the tree, we can just have tree be the size of the witness array for each block, and then call duel repeatedly on every two indices until one is left. Similar to what we did with the vector sum. 
 
	__shared__ int tree[1024]; //Doesn't work with extern.......
	int ID = threadIdx.x + blockIdx.x*wsize;
	tree[threadIdx.x] = ID; 
	
	__syncthreads(); 

	if (threadIdx.x >= 1024){printf("SHARED ERROR\n");}
	for (int s = 1; s < wsize; s = 2*s){
		if ((2*s*threadIdx.x + s < wsize)&&(2*s*threadIdx.x + s + blockIdx.x*wsize < tsize)){ 
			int i = tree[2*s*threadIdx.x];
			int j = tree[2*s*threadIdx.x + s];
			tree[2*s*threadIdx.x] = duel(T, P, W, i, j);
		}
		__syncthreads(); 
	}

	//Now for each block, check the one location where the pattern may possibly occur using brute force. Use thread 0 to finish. 
	if (threadIdx.x == 0){
		int i = 0;
		int m = tree[0]; 
		while (i < psize){
			if (i + m >= tsize){
				m = -1; //Out of bounds
				break;
			}		
			if (i+m>= tsize){printf("OoB error\n");}
			if (T[i+m] != P[i]){
				m = -1;
				break;
			}
			i++; 
		}

		//Store the result
		if (blockIdx.x >= tsize){printf("SHARED ERROR\n");}
		match[blockIdx.x] = m;
	}
}

#endif

