#include <stdio.h>

#ifndef SER
#define SER

__host__ void serialSearch_naive(char*, int, char*, int, int*);
__host__ void KMP(int* F, char* p, int psize, char* t, int tsize, int* match); 
__host__ void failure(int* F, char* P, int psize); 

#endif

#ifndef SER_DEF
#define SER_DEF

//Naive serial Search ******************************************************************
__host__ void serialSearch_naive(char* p, int psize, char* t, int tsize, int* match){
	int iT = 0; //Index in text
	int iP = 0; //Index in pattern
	int iR = 1; //Index in match
	while (iT <= tsize-psize){
		if (t[iT] == p[iP]){
			int found = 1;
			while (iP < psize){
				if (t[iT + iP] != p[iP]){
					found = 0;
					break;				
				}
				iP += 1; 
			}
			if (found){
				match[iR] = iT;
				iR += 1; 
			}
		}
		iP = 0; 
		iT += 1; 
	}
	match[0] = iR-1; 
}


//Optimal Serial Search ******************************************************
__host__ void failure(int* F, char* P, int psize){
	//Compute the failure function
	int k=1;
	int j=0;
	F[0] = -1; 
	while (k < psize){
		if (P[k] == P[j]){
			F[k] = F[j];
		} else {
			F[k] = j;
			j = F[j];
			while ((j >= 0)&&(P[k] != P[j])){
				j = F[j];
			}
		}
		k += 1;
		j += 1;
	}  
}
__host__ void KMP(int* F, char* P, int psize, char* T, int tsize, int* match){
	//Complete the search
	int j = 0;
	int k = 0;
	int iM = 1;
	while (k < tsize){
		if (T[k] == P[j]){ 
			j += 1;
			k += 1; 
			if (j == psize){
				match[iM] = k-j;
				iM += 1;
				j = F[j];
			}
		} else {
			k = F[k];
			if (k < 0){
				j += 1;
				k += 1;
			}
		}
	}
	match[0] = iM-1;
}

#endif
