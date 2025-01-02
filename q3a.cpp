#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <emmintrin.h>
#include <limits.h>
#include <pmmintrin.h>
#include <immintrin.h>

//function declarations
void Gaussian_Blur();
void Sobel();
void read_image(const char* filename);
void write_image2(const char* filename, unsigned char* output_image);
void openfile(const char* filename, FILE** finput);
int getint(FILE* fp);
void read_image_dimensions(const char* filename);
void cleanup_memory(void);
int manage_memory(int required_size);

//CRITICAL POINT: images' paths - You need to change these paths
#define IN "C:\\Users\\acer\\Desktop\\input_images\\"
#define OUT "C:\\Users\\acer\\Desktop\\output_images\\"
#define OUT2 "C:\\Users\\acer\\Desktop\\output_images\\"

// Image dimensions (will be updated dynamically)
int M, N;

int current_allocated_size = 0;

// Arrays to be dynamically allocated 
unsigned char* frame1;
unsigned char* filt;
unsigned char* gradient;


const signed char Mask[5][5] = {//2d gaussian mask with integers
	{2,4,5,4,2} ,
	{4,9,12,9,4},
	{5,12,15,12,5},
	{4,9,12,9,4},
	{2,4,5,4,2}
};

const signed char GxMask[3][3] = {
	{-1,0,1} ,
	{-2,0,2},
	{-1,0,1}
};

const signed char GyMask[3][3] = {
	{-1,-2,-1} ,
	{0,0,0},
	{1,2,1}
};

char header[100];
errno_t err;


int main() {
	char input_filename[256];
	char output_filename_blur[256];
	char output_filename_edge[256];

	printf("Starting image processing.......\n");

	//initializing pointers to null
	frame1 = filt = gradient = NULL;

	for (int i = 0; i <= 30; i++) {
		// generating the file paths
		sprintf_s(input_filename, 256, "%sa%d.pgm", IN, i);
		sprintf_s(output_filename_blur, 256, "%sblurred_%d.pgm", OUT, i);
		sprintf_s(output_filename_edge, 256, "%sedge_%d.pgm", OUT2, i);

		printf("\nProcessing image %d of 31", i + 1);

		// Read image dimensions 
		read_image_dimensions(input_filename);
		printf("\nDimensions: %dx%d", M, N);

		//memory allocation
		int required_size = M * N;
		frame1 = (unsigned char*)realloc(frame1, required_size * sizeof(unsigned char));
		filt = (unsigned char*)realloc(filt, required_size * sizeof(unsigned char));
		gradient = (unsigned char*)realloc(gradient, required_size * sizeof(unsigned char));

		if (!frame1 || !filt || !gradient) {
			printf("Memory allocation failed!\n");
			exit(EXIT_FAILURE);
		}

		//Process the image
		read_image(input_filename);//read image from disc
		Gaussian_Blur(); //blur the image (reduce noise)
		Sobel(); //apply edge detection

		// write output images
		write_image2(output_filename_blur, filt); //store output image to the disc
		write_image2(output_filename_edge, gradient); //store output image to the disc

		printf("Done processing image %d\n", i);
	}
	//cleaning up at the end
	cleanup_memory();
	printf("\nAll images processed successfully!\n");
	return 0;
}

void cleanup_memory() {
	free (frame1);
	free (filt);
	free (gradient);
}


void Gaussian_Blur() {

	int row, col, rowOffset, colOffset;
	int newPixel;
	unsigned char pix;
	//const unsigned short int size=filter_size/2;
	const unsigned short int size = 2;

	/*---------------------- Gaussian Blur ---------------------------------*/
	for (row = 0; row < N; row++) {
		for (col = 0; col < M; col++) {
			newPixel = 0;
			for (rowOffset = -size; rowOffset <= size; rowOffset++) {
				for (colOffset = -size; colOffset <= size; colOffset++) {

					if ((row + rowOffset < 0) || (row + rowOffset >= N) || (col + colOffset < 0) || (col + colOffset >= M))
						pix = 0;
					else
						pix = frame1[M * (row + rowOffset) + col + colOffset];

					newPixel += pix * Mask[size + rowOffset][size + colOffset];

				}
			}
			filt[M * row + col] = (unsigned char)(newPixel / 159);

		}
	}

}


void Sobel() {

	int row, col, rowOffset, colOffset;
	int Gx, Gy;

	/*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			Gx = 0;
			Gy = 0;

			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			for (rowOffset = -1; rowOffset <= 1; rowOffset++) {
				for (colOffset = -1; colOffset <= 1; colOffset++) {

					Gx += filt[M * (row + rowOffset) + col + colOffset] * GxMask[rowOffset + 1][colOffset + 1];
					Gy += filt[M * (row + rowOffset) + col + colOffset] * GyMask[rowOffset + 1][colOffset + 1];
				}
			}

			gradient[M * row + col] = (unsigned char)sqrt(Gx * Gx + Gy * Gy); /* Calculate gradient strength		*/
			//gradient[row][col] = abs(Gx) + abs(Gy); // this is an optimized version of the above

		}
	}


}


void read_image_dimensions(const char* filename) {
	FILE* finput = NULL;
	openfile(filename, &finput);

	fclose(finput);
}


void read_image(const char* filename)
{
	int c;
	FILE* finput;
	int i, j, temp;

	printf("\nReading %s image from disk ...", filename);
	finput = NULL;
	openfile(filename, &finput);

	if ((header[0] == 'P') && (header[1] == '5')) { //if P5 image

		for (j = 0; j < N; j++) {
			for (i = 0; i < M; i++) {

				//if (fscanf_s(finput, "%d", &temp,20) == EOF)
				//	exit(EXIT_FAILURE);
				temp = getc(finput);

				frame1[M * j + i] = (unsigned char)temp;
			}
		}
	}
	else if ((header[0] == 'P') && (header[1] == '2')) { //if P2 image
		for (j = 0; j < N; j++) {
			for (i = 0; i < M; i++) {

				if (fscanf_s(finput, "%d", &temp, 20) == EOF)
					exit(EXIT_FAILURE);

				frame1[M * j + i] = (unsigned char)temp;
			}
		}
	}
	else {
		printf("\nproblem with reading the image");
		exit(EXIT_FAILURE);
	}

	fclose(finput);
	printf("\nimage successfully read from disc\n");

}



void write_image2(const char* filename, unsigned char* output_image)
{

	FILE* foutput;
	int i, j;



	printf("  Writing result to disk ...\n");

	if ((err = fopen_s(&foutput, filename, "wb")) != NULL) {
		fprintf(stderr, "Unable to open file %s for writing\n", filename);
		exit(-1);
	}

	fprintf(foutput, "P2\n");
	fprintf(foutput, "%d %d\n", M, N);
	fprintf(foutput, "%d\n", 255);

	for (j = 0; j < N; ++j) {
		for (i = 0; i < M; ++i) {
			fprintf(foutput, "%3d ", output_image[M * j + i]);
			if (i % 32 == 31) fprintf(foutput, "\n");
		}
		if (M % 32 != 0) fprintf(foutput, "\n");
	}
	fclose(foutput);


}





void openfile(const char* filename, FILE** finput)
{
	int x0, y0, x, aa;

	if ((err = fopen_s(finput, filename, "rb")) != NULL) {
		fprintf(stderr, "Unable to open file %s for reading\n", filename);
		exit(-1);
	}

	aa = fscanf_s(*finput, "%s", header, 20);

	x0 = getint(*finput);//this is M
	y0 = getint(*finput);//this is N
	printf("\t header is %s, while x=%d,y=%d", header, x0, y0);


	M = x0;
	N = y0;


	x = getint(*finput); /* read and throw away the range info */
	//printf("\n range info is %d",x);

}



//CRITICAL POINT: you can define your routines here that create the arrays dynamically; now, the arrays are defined statically.



int getint(FILE* fp) /* adapted from "xv" source code */
{
	int c, i, firstchar;//, garbage;

	/* note:  if it sees a '#' character, all characters from there to end of
	   line are appended to the comment string */

	   /* skip forward to start of next number */
	c = getc(fp);
	while (1) {
		/* eat comments */
		if (c == '#') {
			/* if we're at a comment, read to end of line */
			char cmt[256], * sp;

			sp = cmt;  firstchar = 1;
			while (1) {
				c = getc(fp);
				if (firstchar && c == ' ') firstchar = 0;  /* lop off 1 sp after # */
				else {
					if (c == '\n' || c == EOF) break;
					if ((sp - cmt) < 250) *sp++ = c;
				}
			}
			*sp++ = '\n';
			*sp = '\0';
		}

		if (c == EOF) return 0;
		if (c >= '0' && c <= '9') break;   /* we've found what we were looking for */

		/* see if we are getting garbage (non-whitespace) */
	   // if (c!=' ' && c!='\t' && c!='\r' && c!='\n' && c!=',')
		//	garbage=1;

		c = getc(fp);
	}

	/* we're at the start of a number, continue until we hit a non-number */
	i = 0;
	while (1) {
		i = (i * 10) + (c - '0');
		c = getc(fp);
		if (c == EOF) return i;
		if (c < '0' || c>'9') break;
	}
	return i;
}
