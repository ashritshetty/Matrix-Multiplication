#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void MatrixMultiplyI(int *matrix1, int *matrix2, int *matrix3, int m, int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < m && y < n)
	{
    int offset = ????;
		matrix3[offset] = matrix1[offset] + matrix2[offset];
	}
}

__global__ void MatrixMultiplyF(float *matrix1, float *matrix2, float *matrix3, int m, int n)
{
	__shared__ float ds_M [BLOCKSIZE][BLOCKSIZE];
	__shared__ float ds_N [BLOCKSIZE][BLOCKSIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	float sum = 0.0;

	for (int i = 0; i < n/BLOCKSIZE; n++)
	{
		ds_M [ty][tx] = 
	}
}

void read_imatrix(char *filename, int *m, int *n, int **values)
{
	FILE* name;
	int i, j, k;
	int t1, t2, t3;
	name = fopen(filename, "r+");
	if(name != NULL)
	{
		k = 0;
	  fscanf(name, "%d %d\n", &t1, &t2);
		*m = t1;
		*n = t2;
		*values = (int *)calloc(t1*t2, sizeof(int));
	  for(i = 1; i <= t1; i++)
	  {
	    for(j = 1; j <= t2; j++)
	    {
				if(j < t2)
				{
		  		fscanf(name, "%d,", &t3);
					*(*values+k) = t3;
		  		k++;
				}
				else
				{
		  		fscanf(name, "%d\n", &t3);
					*(*values+k) = t3;
		  		k++;
				}
			}
		}
	  fclose(name);
	}
	else
	{
	  printf("File read failed\n");
	}
}

void read_fmatrix(char *filename, int *m, int *n, float **values)
{
	FILE* name;
	int i, j, k;
	int t1, t2;
	float t3;
	name = fopen(filename, "r+");
	if(name != NULL)
	{
		k = 0;
	  fscanf(name, "%d %d\n", &t1, &t2);
		*m = t1;
		*n = t2;
		*values = (float *)calloc(t1*t2, sizeof(float));
  	for(i = 1; i <= t1; i++)
    {
    	for(j = 1; j <= t2; j++)
      {
				if(j < t2)
				{
	  			fscanf(name, "%f,", &t3);
					*(*values+k) = t3;
	  			k++;
				}
				else
				{
	  			fscanf(name, "%f\n", &t3);
					*(*values+k) = t3;
	  			k++;
				}
			}
		}
    fclose(name);
  }
  else
  {
    printf("File read failed\n");
  }
}

void write_imatrix(char *filename, int *m, int *n, int **values)
{
	FILE* name;
	int i, j, k;
	int t1, t2, t3;
	name = fopen(filename, "w+");
	if(name != NULL)
	{
		k = 0;
		t1 = *m;
		t2 = *n;
	  fprintf(name, "%d %d\n", t1, t2);
    for(i = 1; i <= t1; i++)
    {
      for(j = 1; j <= t2; j++)
      {
				if(j < t2)
				{
					t3 = *(*values+k);
		  		fprintf(name, "%d,", t3);
		  		k++;
				}
				else
				{
					t3 = *(*values+k);
		  		fprintf(name, "%d\n", t3);
		  		k++;
				}
			}
    }
    fclose(name);
  }
  else
  {
    printf("File write failed\n");
  }
}

void write_fmatrix(char *filename, int *m, int *n, float **values)
{
	FILE* name;
	int i, j, k;
	int t1, t2;
	float t3;
	name = fopen(filename, "w+");
	if(name != NULL)
	{
		k = 0;
		t1 = *m;
		t2 = *n;
	  fprintf(name, "%d %d\n", t1, t2);
    for(i = 1; i <= t1; i++)
    {
      for(j = 1; j <= t2; j++)
      {
				if(j < t2)
				{
					t3 = *(*values+k);
		  		fprintf(name, "%f,", t3);
		  		k++;
				}
				else
				{
					t3 = *(*values+k);
		  		fprintf(name, "%f\n", t3);
		  		k++;
				}
			}
    }
    fclose(name);
  }
  else
  {
    printf("File write failed\n");
  }
}

void matrix_check(int m1, int n1, int m2, int n2)
{
  if ((n1-m2) != 0)
  {
    printf("Matrix dimensions must be PxQ and QxR respectively\n");
    exit(1);
  }
}

int main(int argc, char *argv[])
{
  int m1, n1, m2, n2, matrix_size;

	if (argc != 5)
	{
		printf("Usage: ./matrix-multiplication matrix1.mat matrix2.mat matrix3.mat float/int \n");
		exit(1);
	}

	if (strcmp(argv[4], "float") == 0)
  {
		float *hostmatrix1, *hostmatrix2, *hostmatrix3;
		float *devicematrix1, *devicematrix2, *devicematrix3;
  	read_matrix(argv[1], &m1, &n1, &hostmatrix1);
		read_matrix(argv[2], &m2, &n2, &hostmatrix2);
		matrix_check(m1, n1, m2, n2);
		matrix_size = m1 * n1;
  	hostmatrix3 = (float *)calloc(matrix_size, sizeof(float));
		cudaMalloc(&devicematrix1, matrix_size);
  	cudaMalloc(&devicematrix2, matrix_size);
  	cudaMalloc(&devicematrix3, matrix_size);
		cudaMemcpy(devicematrix1, hostmatrix1, matrix_size, cudaMemcpyHostToDevice);
  	cudaMemcpy(devicematrix2, hostmatrix2, matrix_size, cudaMemcpyHostToDevice);
		dim3 dimGrid(m1/BLOCKSIZE, n1/BLOCKSIZE, 1);
		dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
		MatrixMultiplyF <<< dimGrid, dimBlock >>> (devicematrix1, devicematrix2, devicematrix3, m1, n1);
		cudaMemcpy(hostmatrix3, devicematrix3, matrix_size, cudaMemcpyDeviceToHost);
  	write_matrix(argv[3], &m1, &n1, &hostmatrix3);
		cudaFree(devicematrix1);
  	cudaFree(devicematrix2);
  	cudaFree(devicematrix3)
		free(hostmatrix1);
  	free(hostmatrix2);
  	free(hostmatrix3);
	}

	if (strcmp(argv[4], "int") == 0)
  {
		int *hostmatrix1, *hostmatrix2, *hostmatrix3;
		int *devicematrix1, *devicematrix2, *devicematrix3;
  	read_matrix(argv[1], &m1, &n1, &hostmatrix1);
		read_matrix(argv[2], &m2, &n2, &hostmatrix2);
		matrix_check(m1, n1, m2, n2);
		matrix_size = m1 * n1;
  	hostmatrix3 = (int *)calloc(matrix_size, sizeof(int));
		cudaMalloc(&devicematrix1, matrix_size);
  	cudaMalloc(&devicematrix2, matrix_size);
  	cudaMalloc(&devicematrix3, matrix_size);
		cudaMemcpy(devicematrix1, hostmatrix1, matrix_size, cudaMemcpyHostToDevice);
  	cudaMemcpy(devicematrix2, hostmatrix2, matrix_size, cudaMemcpyHostToDevice);
		dim3 dimGrid(matrix_size/BLOCKSIZE, 1, 1);
		dim3 dimBlock(BLOCKSIZE, 1, 1);
		MatrixMultiplyI <<< dimGrid, dimBlock >>> (devicematrix1, devicematrix2, devicematrix3, m1, n1);
		cudaMemcpy(hostmatrix3, devicematrix3, matrix_size, cudaMemcpyDeviceToHost);
  	write_matrix(argv[3], &m1, &n1, &hostmatrix3);
		cudaFree(devicematrix1);
  	cudaFree(devicematrix2);
  	cudaFree(devicematrix3)
		free(hostmatrix1);
  	free(hostmatrix2);
  	free(hostmatrix3);
	}

  return 0;
}
