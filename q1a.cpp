#include <stdio.h> //this library is needed for printf function
#include <stdlib.h> //this library is needed for rand() function
#include <windows.h> //this library is needed for pause() function
#include <time.h> //this library is needed for clock() function
#include <math.h> //this library is needed for abs()
#include <omp.h> //this library is needed for the timer
#include <immintrin.h> // this library is needed for AVX intrinsics

void init();
void q1();
void q1_vec_j();
void q1_vec_k();
void check_correctness();

#define EPSILON 1e-5 //relative error margin

#define N 256 //input size
float A[N][N], B[N][N], C[N][N], C_vec[N][N];

#define TIMES_TO_RUN 1 //how many times the function will run. If the ex_time you get is lower than 2 seconds, then increase this value accordingly



int main() {

	//define the timers measuring execution time
	double start_1, end_1;
	//clock_t start_1, end_1; //ignore this for  now

	init();//initialize the arrays

	 
	start_1 = omp_get_wtime(); //start the timer 
	//start_1 = clock(); //start the timer 
	for (int i = 0; i < TIMES_TO_RUN; i++) { // NOTE: IF you use this loop to increase time, output doesnot match result will be displayed.
		q1(); //original main routine
	}
	end_1 = omp_get_wtime(); //end the timer 
	//end_1 = clock(); //end the timer 
	double ex_time = (end_1 - start_1) / TIMES_TO_RUN; //calculating the average execution time

	printf("Original q1 time in seconds: %f\n", end_1 - start_1);//print the ex.time
	//printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));//print the ex.time
	

	start_1 = omp_get_wtime(); //start the timer for vectorized j loop
	//start_1 = clock(); //start the timer 
	q1_vec_j();
	end_1 = omp_get_wtime(); //end the timer 
	//end_1 = clock(); //end the timer
	printf("Vectorized q1_vec_j time in seconds: %f\n", end_1 - start_1); //print the ex.time
	//printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));//print the ex.time
	
	start_1 = omp_get_wtime(); //start the timer for vectorized k loop
	//start_1 = clock(); //start the timer
	q1_vec_k();
	end_1 = omp_get_wtime(); //end the timer 
	//end_1 = clock(); //end the timer
	printf("Vectorized q1_vec_k time in seconds: %f\n", end_1 - start_1); //print the ex.time
	//printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));//print the ex.time
	
	

	//performing the correctness check
	check_correctness();

	// FLOPS calculation for q1() routine....
	double flops = (2.0 * N * N * N) / ex_time; // using flops calculation formula
	flops = flops / 1e9; // converting the flops value to Giga FLOPS
	printf("The FLOPS value in Giga FLOPS is  %f\n", flops); //print the FLOPS value
	
	

	system("pause"); //this command does not let the output window to close

	return 0; //normally, by returning zero, we mean that the program ended successfully. 
}



void init() {

	float e = 0.1234f, p = 0.7264f;//if you do not specify the 'f' after 0.0, then double precision data type is assumed (not float which single precision). 

	//MVM
	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < N; j++) {
			A[i][j] = ((i - j) % 9) + p;
			B[i][j] = ((i + j) % 11) + e;
			C[i][j] = 0.0f;
			C_vec[i][j] = 0.0f; // initializing C_vec for vectorized routines
		}

}



void q1() {

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void q1_vec_j() { //task_B

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j += 8) { // process 8 elements at a time
			__m256 c_vec = _mm256_setzero_ps(); // initializing the result vector to zero
			for (int k = 0; k < N; k++) {
				__m256 a_vec = _mm256_set1_ps(A[i][k]); // setting A[i][k] to all elements
				__m256 b_vec = _mm256_loadu_ps(&B[k][j]); // loading 8 elements from B[k]
				c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec); // multiplying and adding
			}
			_mm256_storeu_ps(&C_vec[i][j], c_vec); // storing the result back to C_vec
		}
	}
}

void q1_vec_k() { //task_C
	// Allocate memory for B_transposed on the heap
	float** B_transposed = (float**)malloc(N * sizeof(float*));
	for (int i = 0; i < N; i++) {
		B_transposed[i] = (float*)malloc(N * sizeof(float));
	}

	// Transposing B for better memory access
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			B_transposed[i][j] = B[j][i];
		}
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			__m256 c_vec = _mm256_setzero_ps(); // initializing the result vector to zero

			for (int k = 0; k < N; k += 4) { // processing 8 elements of B_transposed at a time
				__m256 a_vec = _mm256_loadu_ps(&A[i][k]); //loading 8 elements from A[i]
				__m256 b_vec = _mm256_loadu_ps(&B_transposed[j][k]); // loading 8 elements from transposed B
				c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec); // multiplying and adding
			}

			c_vec = _mm256_hadd_ps(c_vec, c_vec); // horizontal addition
			c_vec = _mm256_hadd_ps(c_vec, c_vec);

			_mm256_storeu_ps(&C_vec[i][j], c_vec); // storing the result back to C_vec
		}
	}

	// Free the allocated memory
	for (int i = 0; i < N; i++) {
		free(B_transposed[i]);
	}
	free(B_transposed);
}





void check_correctness() {
	int correct = 1;
	const float epsilon = 1e-5f; 
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (fabs(C[i][j] - C_vec[i][j])/fabs(C[i][j]) > epsilon) {
				correct = 0;
			}
		}
	}
	if (correct) {
		printf("Outputs match.\n");
	}
	else {
		printf("Outputs do not match.\n");
	}
}
