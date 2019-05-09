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
	int tsize = 2048*2048*256; //largest int allowed
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

	warmup<<<1,1>>>(); //Warmup the GPU to ensure accurrate timing


	//Now, let's run tests on the GPU versions and compare to CPU results
	//******************************************************************** 

	//Test: synced parallel version w/global tree and cpu witness
	FILE *data = fopen("Synced.txt", "w");
	test(text, tsize, 2, 1, data);
	printf("\n\n"); 
	fflush(stdout); 
	fclose(data);

	//Test: synced parallel version w/shared tree and cpu witness
	data = fopen("SyncedShared.txt", "w");
	test(text, tsize, 2, 2, data); 
	printf("\n\n"); 
	fflush(stdout);//
	fclose(data);

	//Test: multiple kernel version and cpu witness
	data = fopen("Multiple.txt", "w");
	test(text, tsize, 2, 3, data); 
	printf("\n\n"); 
	fflush(stdout); 
	fclose(data); 
}
