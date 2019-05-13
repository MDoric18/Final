//Marina Doric
//Final Project

#include <stdio.h>
#include "timerc.h"
#include "test.cuh"
#include "witness.cuh"
#include "serial.cuh"

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
			i += 1;		
		}
		c = getchar();  
	}

	//If the text was smaller than the ascribed size, reset.
	tsize = i; 

	warmup<<<1,1>>>(); //Warmup the GPU to ensure accurrate timing
	gerror(cudaPeekAtLastError()); 


	//Now, let's run tests on the GPU versions and compare to CPU results
	//******************************************************************** 

	FILE *data;

	int size = 2048*32; //Constraint of constant memory version
	int maxp = size; 
	int inc = 50; //Large to speed up data collection

	//Test: Multiple Kernel Constant Mem on largest patterns possible
	data = fopen("Constant.txt", "w");
	test(text, size, 2, 4, data, maxp, inc);
	printf("\n\n");
	fflush(stdout);
	fclose(data); //*/ 

	//FIRST, Generic comparisons and tests on limited text for all four versions
	//Test: synced parallel version w/global tree and cpu witness
	maxp = 2048; //Constraint of synced versions
	inc = 4; //Do pattern sizes 4,8,12,16,...,2048
	data = fopen("Synced_WitCPU.txt", "w");
	test(text, size, 2, 1, data, maxp, inc);
	printf("\n\n"); 
	fflush(stdout); 
	fclose(data);//*/

	//Test: synced parallel version w/shared tree and cpu witness
	data = fopen("SyncedShared_WitCPU.txt", "w");
	test(text, tsize, 2, 2, data, maxp, inc); 
	printf("\n\n"); 
	fflush(stdout);
	fclose(data);//*/

	//Test: synced parallel version w/global tree and gpu witness
	data = fopen("Synced_WitGPU.txt", "w");
	test(text, tsize, 1, 1, data, maxp, inc);
	printf("\n\n"); 
	fflush(stdout); 
	fclose(data);//*/

	//Test: synced parallel version w/shared tree and gpu witness
	data = fopen("SyncedShared_WitGPU.txt", "w");
	test(text, tsize, 1, 2, data, maxp, inc); 
	printf("\n\n"); 
	fflush(stdout);//
	fclose(data);//*/

	//Test: multiple kernel version and cpu witness
	data = fopen("Multiple_WitCPU.txt", "w");
	test(text, tsize, 2, 3, data, maxp, inc); 
	printf("\n\n"); 
	fflush(stdout); 
	fclose(data);//*/
	
	//Test: multiple kernel version and gpu witness
	data = fopen("Multiple_WitGPU.txt", "w");
	test(text, tsize, 1, 3, data, maxp, inc); 
	printf("\n\n"); 
	fflush(stdout); 
	fclose(data); //*/

	//Test: multiple kernel version and cpu witness
	//pattern sizes 8,16,24,32...2048
	data = fopen("MultipleConstant_WitCPU.txt", "w");
	test(text, tsize, 2, 4, data, maxp, inc); 
	printf("\n\n"); 
	fflush(stdout); 
	fclose(data);//*/
	
	//Test: multiple kernel version and gpu witness
	//pattern sizes 8,16,24,32...2048
	data = fopen("MultipleConstant_WitGPU.txt", "w");
	test(text, tsize, 1, 4, data, maxp, inc); 
	printf("\n\n"); 
	fflush(stdout); 
	fclose(data);//*/


	//Did generate this data below but didn't talk about it
	//SECOND, Compare long term Multiple kernel version and serial version on whole text with largest pattern sizes possible
	//The multiple kernel version encounters an illegal memory access when the pattern sizes is 65537 or larger. So, we will restrict it.
	maxp = 65537;
	inc = 50; //To keep the run time of testing down
	size = tsize; //Run on full thing 

	//Test: multiple kernel version and cpu witness
	data = fopen("Multiple_WitCPU_WHOLE.txt", "w");
	test(text, tsize, 2, 3, data, maxp, inc); 
	printf("\n\n"); 
	fflush(stdout); 
	fclose(data);//*/
	
	//Test: multiple kernel version and gpu witness
	data = fopen("Multiple_WitGPU_WHOLE.txt", "w");
	test(text, tsize, 1, 3, data, maxp, inc); 
	printf("\n\n"); 
	fflush(stdout); 
	fclose(data);//*/
}
