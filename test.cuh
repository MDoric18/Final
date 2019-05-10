#include <stdio.h>
#include "timerc.h"
#include "synced.cuh"
#include "multiple.cuh"
#include "serial.cuh"
#include "witness.cuh"
#include <time.h>

#ifndef TEST
#define TEST

__host__ void test(char*, int, int, int, FILE*); 
__host__ void run(int*, int, char*, int, char*, int, int*, int*, int, int); 
__host__ void printTiming(int, int);
__global__ void warmup();

#endif

#ifndef TEST_CODE
#define TEST_CODE

float copy1 = 0;
float copy1b = 0; 
float copy2 = 0;
float witTime = 0;
float time1 = 0;
float time2 = 0;
float clean = 0;
float ser = 0; 
float kmp = 0; 
float fail = 0; 
int ind; 

//COPIED FROM CLASS CODE
#define gerror(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
//END COPY

__global__ void warmup(){}

__host__ void test(char* T, int tsize, int flag_W, int flag_Func, FILE* data){
	int size = tsize; //4096*1024; 
	printf("Testing with text size %d, and pattern sizes 1:1/2*size+1\n", size);
	if(flag_W == 1){
		printf("\tUsing GPU witness function.\n"); 
	}
	if(flag_W == 2){
		printf("\tUsing CPU witness function.\n");
	}
	if(flag_Func == 1){
		printf("\tUsing Synced Version w/o Shared mem Search Alg.\n");
		fputs("Pattern Size;Naive Serial;Optimal Serial;Synced Global\n", data);   
	}
	if(flag_Func == 2){
		printf("\tUsing Synced Version w/ Shared mem Search Alg.\n");
		fputs("Pattern Size;Naive Serial;Optimal Serial;Synced Shared\n", data);   
	}
	if(flag_Func == 3){
		printf("\tUsing Multiple Kernel Version Search Alg. \n");
		fputs("Pattern Size;Naive Serial;Optimal Serial;Multiple\n", data);   
	}

	//Setup memory for gpu version
	//*****************************************************
	char* devP;
	char* devT;
	int* devR;
	int* devW; 
	int* devTree;
	int* results = (int*) malloc(size*sizeof(int));
	int* matchGPU = (int*) malloc(size*sizeof(int));
	int* matchKMP = (int*) malloc(size*sizeof(int)); 
	int* match = (int*) malloc(size*sizeof(int)); 
	int* cleanArr = (int*) malloc(size*sizeof(int));
	int* F = (int*) malloc(size*sizeof(int)); 
	for (int i=0; i<size; i++){
		cleanArr[i] = -1; 
	}

	float temp;
	gstart();
	gerror(cudaMalloc((void**) &devP, size*sizeof(char)));
	gerror(cudaMalloc((void**) &devT, size*sizeof(char)));
	gerror(cudaMalloc((void**) &devR, size*sizeof(int)));
	gerror(cudaMalloc((void**) &devW, size*sizeof(int))); 
	gerror(cudaMalloc((void**) &devTree, size*sizeof(int))); 
	gerror(cudaMemcpy(devT, T, size*sizeof(char), cudaMemcpyHostToDevice));
	gend(&temp);
	copy1 += temp; 

	//Now test, psize must be <=2048 because of synced version thread restraints
	int success = 1;  
	//Prep phrase for timing reports
	printf("The average runtimes in 1000 iterations:\n"); 
	//Can't run larger than 2048 because of thread constraints
	for (int psize=1000; psize <= 2048; psize += 10){ //Can't run with bigger pattern.  
		int wsize = 0;
		for (int iter=0; iter < 1; iter++){
			//Pick a random location in T to be P with no period
			//int ind;
			char* P;

			srand(time(NULL));
			ind = rand() % (size - psize); 
			P = T + ind;
			if (ind >= size - psize){printf("IND ERROR\n");}

			//Run the serial algorithm
			cstart();
			serialSearch_naive(P, psize, T, size, match);
			cend(&temp);
			ser += temp; 

			//Run the optimal algorithm
			cstart();
			failure(F, P, psize);
			cend(&temp);
			fail += temp;

			/*cstart();
			KMP(F, P, psize, T, size, matchKMP);
			cend(&temp);
			kmp += temp; 

			//Verify the optimal algorithm is working
			for (int k=0; k<match[0]; k++){
				if (match[k+1] != matchKMP[k+1]){
					printf("KMP FAILED\n");
					break; 
				}
			}//*/

			//Calculate wsize
			wsize = psize/2;
			if (psize % 2 == 1){ wsize += 1;}

			gstart();
			gerror(cudaMemcpy(devP, P, psize*sizeof(char), cudaMemcpyHostToDevice)); 
			gend(&temp); 
			copy1b += temp;

			//Clear the result array to ensure accurate copying
			gerror(cudaMemcpy(devR, cleanArr, size*sizeof(int), cudaMemcpyHostToDevice));

			//Now run
			run(devW, wsize, devP, psize, devT, size, devR, devTree, flag_W, flag_Func);

			//Copy back results
			gstart();
			gerror(cudaMemcpy(matchGPU, devR, size*sizeof(int), cudaMemcpyDeviceToHost));
			gend(&temp);
			copy2 += temp; 

			//Now clean up the results (remove -1s and extra 0s)
			int count = 0;
			int pastZero = 0; 
			cstart(); 
			for (int i=0; i < size; i++){
				if(pastZero){
					if (matchGPU[i] > 0){
						results[count] = matchGPU[i];
						count += 1;
					}
				}else {
					if (matchGPU[i] >= 0){
						results[count] = matchGPU[i];
						pastZero = 1; 
						count += 1; 
					}
				} 
			} 
			cend(&temp); 
			clean += temp; 

			//Now compare results to CPU
			int end = 0;
			for (int i=0; i < match[0]; i++){ 
				if (results[i] != match[i+1]){
					printf("Mismatch: %d(GPU) != %d(CPU)\n", results[i], match[i+1]); 
					success = 0; 
					printf("\tERROR, MISMATCH, psize=%d, wsize=%d, loc=%d\n", psize, wsize, ind);
					printf("Count w/ Pattern size = %d: %d (GPU), %d (CPU)\n", psize, count, match[0]);
					int block = size/wsize;
					if (tsize % wsize > 0){block += 1;}
					end = 1;
					break; 

				} 
			}
			if (match[0] != count) { break;}
			if (end == 1) { break;}
		}
		if (success==0){ 
			printf("Failed.\n");
			break;
		}else{
			//Now print information regarding timing
			/*printf("\t\tPattern size: %d\n", psize); 
			printf("\t\tNaive serial time: %f\n", ser/1000);
			printf("\t\tFailure function time: %f\n", fail/1000);
			printf("\t\tKMP search time: %f\n", kmp/1000);
			printf("\t\tTotal KMP time: %f\n", (kmp+fail)/1000);  
			printf("\t\tInitial copy time: %f\n", copy1 + copy1b/1000);
			printf("\t\tCopy results time: %f\n", copy2/1000);
			printf("\t\tClean results time: %f\n", clean/1000); 
			printf("\t\tWitness creation time: %f\n", witTime/1000);
			if (flag_Func < 3){
				printf("\t\tSearch time: %f\n", time1/1000);
				fprintf(data, "%d;%f;%f;%f\n", psize, ser/1000, (kmp+fail)/1000, time1/1000); 
			}
			if (flag_Func == 3){
				printf("\t\tCreate tree time: %f\n", time1/1000);
				printf("\t\tFinish search time: %f\n", time2/1000); 
				fprintf(data, "%d;%f;%f;%f\n", psize, ser/1000, (kmp+fail)/1000, (time1+time2)/1000);
			}
			printf("\t\tTotal time: %f\n\n", copy1 + (copy1b + copy2 + clean + witTime + time1 + time2)/1000);*/
			copy1b = 0;
			copy2 = 0;
			clean = 0;
			witTime = 0; 
			time1 = 0;
			time2 = 0;
			ser = 0; 
			
		}
		cudaDeviceSynchronize(); 
	}
	if (success){ printf("Success!\n");}

	gerror(cudaFree(devP));
	gerror(cudaFree(devT));
	gerror(cudaFree(devR));
	gerror(cudaFree(devW)); 
	gerror(cudaFree(devTree)); 
	free(matchGPU);
	free(matchKMP);
	free(results);
	free(match);
	free(cleanArr);
	free(F);   
}

__host__ void run(int* W, int wsize, char* P, int psize, char* T, int tsize, int* R, int* tree, int flag_W, int flag_Func){ 
	float temp; 
	int* wit = (int *) malloc(sizeof(int)*wsize);
	char* pat = (char *) malloc(sizeof(char)*psize); 
	cudaMemcpy(pat, P, sizeof(char)*psize, cudaMemcpyDeviceToHost); 
	//if flag_W == 1, we get it on the GPU, else get on CPU
	if (flag_W == 1){
		gstart();
		witnessGPU<<<1024,1024>>>(P, psize, W, wsize);
		gend(&temp);
		witTime += temp; 
		cudaDeviceSynchronize();
	} else{
		if (flag_W == 2){
			cstart();
			witnessCPU(pat, psize, wit, wsize); 
			cend(&temp);
			gerror(cudaPeekAtLastError());
			witTime += temp; 
			gerror(cudaMemcpy(W, wit, sizeof(int)*wsize, cudaMemcpyHostToDevice)); 
		}else{ printf("ERROR flag_W\n");} 
	}

	//if flag_Func == 1, use search_synced
	//if flag_Func == 2, use search_synced_shared
	//if flag_Func == 3, use multiple kernel version w/o constant mem 
	if (flag_Func == 1){
		int block = tsize/wsize;
		if (tsize % wsize > 0){block += 1;}
		gstart();
		search_synced<<<block,wsize>>>(W, wsize, P, psize, T, tsize, R, tree);
		gend(&temp);
		gerror(cudaPeekAtLastError());
		time1 += temp; 
		cudaDeviceSynchronize();
		
	} 
	if (flag_Func == 2){
		int block = tsize/wsize;
		if (tsize % wsize > 0){block += 1;}
		gstart();
		search_synced_shared<<<block,wsize>>>(W, wsize, P, psize, T, tsize, R);
		gend(&temp);
		cudaDeviceSynchronize();
		gerror(cudaPeekAtLastError());
		time1 += temp; 
	
	}
	if (flag_Func == 3){
		int div = 1024;
		int block = tsize/div;
		if (tsize % div > 0){block += 1;}
		printf("PRE\n");
		gstart();
		createTreeLevel<<<block,div>>>(W, wsize, P, psize, T, tsize, tree, 0);	
		for (int s = 1; s < wsize; s = 2*s){
			createTreeLevel<<<block, div>>>(W, wsize, P, psize, T, tsize, tree, s);	
		}
		gend(&temp); 
		time1 += temp; 
		cudaDeviceSynchronize();
		gerror(cudaPeekAtLastError());
		printf("MID\n");

		gstart();
		search_Finish<<<block,div>>>(wsize, P, psize, T, tsize, R, tree);
		gend(&temp);
		time2 += temp; 
		printf("Ind:%d\n", ind); 
		fflush(stdout); 
		cudaDeviceSynchronize();
		gerror(cudaPeekAtLastError());
	}
	if ((flag_Func > 3)||(flag_Func < 1)){ 
		printf("ERROR flag_Func\n");
	}
}
#endif
