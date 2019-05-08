//Marina Doric
//Final Project

#include <stdio.h>
#include "timerc.h"
#include "test.cuh"
#include "witness.cuh"
#include "serial.cuh"

//__constant__ char textC[2048*2048*256];

//Main function ***********************************************************************
int main(){
	//Assume this has been run while pulling info from file "DNA.txt"

	//Set up the data
	int tsize = 10000000; //Largest size allowed is 2048*2048*256 but text is only size 10,000,000
	char* text = (char*) malloc(tsize*sizeof(char)); //text

	//Process text
	int i=0;
	char c = getchar();
	while ((i<tsize)&&(c != EOF)){
		if ((c != 13)&&(c != 10)&&(c != 0)){
			text[i] = c;
			//cudaMemcpyToSymbol(&textC[i],&c,sizeof(char)); 
			i += 1;		
		}
		c = getchar();  
	}

	//If the text was smaller than the ascribed size, reset.
	tsize = i; 

/*	//Print times and check results
	printf("\nThe time to find the locations and the matches are as follows: \n\n");
	printf("SERIAL NAIVE CPU: %f\n", cpuSerialTime);
	int wrong = 0;
	for (int i=0; i < match[0]; i++){
		//Verify this really is a match
		for (int j=0; j<psize; j++){
			if (text[match[i+1]+j] != pattern[j]){
				wrong += 1;
				break;
			}
		}
		if (wrong > 0){ printf("ERROR(%d) ", match[i+1]);}
	}
	if (wrong == 0){
		printf("SERIAL NAIVE CPU: No errors reported"); 
	}
	printf("\nSERIAL NAIVE CPU found %d matches.\n\n", match[0]); 

*/

	warmup<<<1,1>>>(); //Warmup the GPU to ensure accurrate timing


	//Now, let's run tests on the GPU versions and compare to CPU results
	//******************************************************************** 

	//Test: synced parallel version w/global tree and cpu witness
	test(text, tsize, 2, 1);
	printf("\n\n"); 
	fflush(stdout); 

	//Test: synced parallel version w/shared tree and cpu witness
	test(text, tsize, 2, 2); 
	printf("\n\n"); 
	fflush(stdout);

	//Test: multiple kernel version and cpu witness
	test(text, tsize, 2, 3); 
	fflush(stdout); 

	return 0; //PUT THIS HERE UNTIL CREATE TEST


/*
	//Get the witness array on GPU
	float witTime;
	gstart();
	witnessGPU<<<256,256>>>(devP, psize, devW, wsize);
	gend(&witTime);
	cudaDeviceSynchronize(); 

	//Next, let's run and time the synced parallel algorithm 
	//*******************************************************************************
	
	//Call the search
	float search_syncedTime;
	gstart();
	search_synced<<<tsize/wsize,wsize>>>(devW, wsize, devP, psize, devT, tsize, devR, devTree); 
	gend(&search_syncedTime);
	cudaDeviceSynchronize(); 

	//Copy back results
	int* matchGPU = (int*) malloc((tsize/wsize)*sizeof(int));
	float copy2;
	gstart();
	cudaMemcpy(matchGPU, devR, (tsize/wsize)*sizeof(int), cudaMemcpyDeviceToHost); 
	gend(&copy2); 

	//Next, let's run and time the synced parallel algorithm with shared memory
	//********************************************************************************
	//Already got witness array
	
	//Call the search
	float search_synced_sharedTime;
	gstart();
	search_synced_shared<<<tsize/wsize,wsize,wsize*sizeof(int)>>>(devW, wsize, devP, psize, devT, tsize, devR); 
	gend(&search_synced_sharedTime);
	cudaDeviceSynchronize(); 

	//Copy back results
	int* matchGPU3 = (int*) malloc((tsize/wsize)*sizeof(int));
	float copy3;
	gstart();
	cudaMemcpy(matchGPU3, devR, (tsize/wsize)*sizeof(int), cudaMemcpyDeviceToHost); 
	gend(&copy3); 

	//Next, let's run and time the multiple kernel version
	//*******************************************************************

	//The witness array remains untouched in devW, so no need to recopy this.

	//Loop and call kernels to create the duel tree
	int div = 16; //Change this value to test efficiency with diff block/thread counts
	float createTreeTime;
	gstart();
	for (int s = 1; s < wsize; s = 2*s){
		createTreeLevel<<<tsize/div, div>>>(devW, wsize, devP, psize, devT, tsize, devTree, s);	
	}
	gend(&createTreeTime); 

	//Now, check for the pattern
	int div2 = 32; //Change this value to test efficiency with diff block/thread counts
	float finishTime;
	gstart();
	search_Finish<<<tsize/div2, div2>>>(wsize, devP, psize, devT, tsize, devR, devTree); 
	gend(&finishTime);

	//Copy back results
	int* matchGPU2 = (int*) malloc((tsize/wsize)*sizeof(int));
	float copy2b;
	gstart();
	cudaMemcpy(matchGPU2, devR, (tsize/wsize)*sizeof(int), cudaMemcpyDeviceToHost); 
	gend(&copy2b);  

	//Now, let's test the GPU results against the CPU and print time
	//********************************************************************************
	printf("SYNCED GPU: %f (search), %f (+copy times)\n", search_syncedTime, search_syncedTime+witTime+copy1+copy2); 
	int count = 0;
	wrong = 0; 
	for (int i=0; i < tsize/wsize; i++){
		if (matchGPU[i] >= 0){
			//Verify this really is a match
			for (int j=0; j<psize; j++){
				if (text[matchGPU[i]+j] != pattern[j]){
					wrong += 1; 
				}	
			}
			if (wrong > 0){ printf("\nERROR(%d)\n", matchGPU[i]);}
			wrong = 0;
			count += 1; 
		}
	}
	printf("\nSYNCED GPU found %d matches.\n\n", count); 

	//Now, let's test the GPU results against the CPU and print time
	//********************************************************************************
	printf("SYNCED GPU w/SHARED: %f (search), %f (+copy times)\n", search_synced_sharedTime, search_synced_sharedTime+witTime+copy1+copy3); 
	count = 0;
	wrong = 0; 
	for (int i=0; i < tsize/wsize; i++){
		if (matchGPU3[i] >= 0){
			//Verify this really is a match
			for (int j=0; j<psize; j++){
				if (text[matchGPU3[i]+j] != pattern[j]){
					wrong += 1; 
				}	
			}
			if (wrong > 0){ printf("\nERROR(%d)\n", matchGPU3[i]);}
			wrong = 0;
			count += 1; 
		}
	}
	printf("\nSYNCED GPU w/SHARED found %d matches.\n\n", count); 

	//Now, let's test the GPU results against the CPU and print time
	//********************************************************************************
	printf("MULTIPLE KERNEL GPU: %f + %f = %f (creating tree + finishing = total), %f (+copy times)\n", createTreeTime, finishTime, createTreeTime + finishTime, createTreeTime + finishTime +witTime+copy1+copy2b); 
	count = 0;
	wrong = 0; 
	for (int i=0; i < tsize/wsize; i++){
		if (matchGPU2[i] >= 0){
			//Verify this really is a match
			for (int j=0; j<psize; j++){
				if (text[matchGPU2[i]+j] != pattern[j]){
					wrong += 1; 
				}	
			}
			if (wrong > 0){ printf("\nERROR(%d)\n", matchGPU2[i]);}
			wrong = 0;
			count += 1; 
		}
	}
	printf("\nMULTIPLE KERNEL GPU found %d matches.\n\n", count); 
	
	test(); 

	cudaDeviceSynchronize(); 
*/
}
