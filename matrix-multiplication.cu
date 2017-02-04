#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 16

__global__ void MatrixMultiplyI(int *matrix1, int *matrix2, int *matrix3, int m1, int n1, int m2, int n2)
{
  int i, j;

  __shared__ int ds_M[BLOCKSIZE][BLOCKSIZE];
  __shared__ int ds_N[BLOCKSIZE][BLOCKSIZE];
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  int sum = 0;
  for (i = 0; i < (n2-1)/BLOCKSIZE+1; i++)
  {
    if(row < n2 && i*BLOCKSIZE+tx < n2)
    {
      ds_M[ty][tx] = matrix1[row*n2 + i*BLOCKSIZE+tx];
    }
    else
    {
      ds_M[ty][tx] = 0;
    }
    if(i*BLOCKSIZE+ty < n2 && col < n2)
    {
      ds_N[ty][tx] = matrix2[(i*BLOCKSIZE+ty)*n2 + col];
    }
    else
    {
      ds_N[ty][tx] = 0;
    }
    __syncthreads();
    if(row < n2 && col < n2)
    {
      for (j = 0; j < BLOCKSIZE; j++)
      {
        sum = sum + (ds_M[ty][j] * ds_N[j][tx]);
      }
    }
    __syncthreads();
  }
  if(row < n2 && col < n2)
    matrix3[row*n2+col] = sum;
}

__global__ void MatrixMultiplyF(float *matrix1, float *matrix2, float *matrix3, int m1, int n1, int m2, int n2)
{
  int i, j;

  __shared__ float ds_M[BLOCKSIZE][BLOCKSIZE];
  __shared__ float ds_N[BLOCKSIZE][BLOCKSIZE];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  float sum = 0.0;

  for (i = 0; i < n2/BLOCKSIZE; i++)
  {
    if(row < n2 && i*BLOCKSIZE+tx < n2)
    {
      ds_M[ty][tx] = matrix1[row*n2 + i*BLOCKSIZE+tx];
    }
    else
    {
      ds_M[ty][tx] = 0.0;
    }
    if(i*BLOCKSIZE+ty < n2 && col < n2)
    {
      ds_N[ty][tx] = matrix2[(i*BLOCKSIZE+ty)*n2 + col];
    }
    else
    {
      ds_N[ty][tx] = 0.0;
    }
    __syncthreads();
    if(row < n2 && col < n2)
    {
      for (j = 0; j < BLOCKSIZE; j++)
      {
        sum = sum + (ds_M[ty][j] * ds_N[j][tx]);
      }
    }
    __syncthreads();
  }
  if(row < n2 && col < n2)
    matrix3[row*n2+col] = sum;
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
    exit(1);
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
    exit(1);
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
    exit(1);
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
    exit(1);
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
  int m1, n1, m2, n2, GRIDX, GRIDY;

  if (argc != 5)
  {
    printf("Usage: ./matrix-multiplication matrix1.mat matrix2.mat matrix3.mat float/int \n");
    exit(1);
  }

  if (strcmp(argv[4], "float") == 0)
  {
    float *hostmatrix1, *hostmatrix2, *hostmatrix3;
    float *devicematrix1, *devicematrix2, *devicematrix3;
    read_fmatrix(argv[1], &m1, &n1, &hostmatrix1);
    read_fmatrix(argv[2], &m2, &n2, &hostmatrix2);
    matrix_check(m1, n1, m2, n2);
    size_t matrix_size = m1*n1*sizeof(float);
    hostmatrix3 = (float *)calloc(m1*n1, sizeof(float));
    cudaMalloc(&devicematrix1, matrix_size);
    cudaMalloc(&devicematrix2, matrix_size);
    cudaMalloc(&devicematrix3, matrix_size);
    cudaMemcpy(devicematrix1, hostmatrix1, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devicematrix2, hostmatrix2, matrix_size, cudaMemcpyHostToDevice);
    GRIDX = (int)ceil((float)m1/BLOCKSIZE);
    GRIDY = (int)ceil((float)n2/BLOCKSIZE);
    dim3 dimGrid(GRIDX, GRIDY, 1);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    MatrixMultiplyF <<< dimGrid, dimBlock >>> (devicematrix1, devicematrix2, devicematrix3, m1, n1, m2, n2);
    cudaMemcpy(hostmatrix3, devicematrix3, matrix_size, cudaMemcpyDeviceToHost);
    write_fmatrix(argv[3], &m1, &n1, &hostmatrix3);
    cudaFree(devicematrix1);
    cudaFree(devicematrix2);
    cudaFree(devicematrix3);
    free(hostmatrix1);
    free(hostmatrix2);
    free(hostmatrix3);
  }

  if (strcmp(argv[4], "int") == 0)
  {
    int *hostmatrix1, *hostmatrix2, *hostmatrix3;
    int *devicematrix1, *devicematrix2, *devicematrix3;
    read_imatrix(argv[1], &m1, &n1, &hostmatrix1);
    read_imatrix(argv[2], &m2, &n2, &hostmatrix2);
    matrix_check(m1, n1, m2, n2);
    size_t matrix_size = m1*n1*sizeof(int);
    hostmatrix3 = (int *)calloc(m1*n1, sizeof(int));
    cudaMalloc(&devicematrix1, matrix_size);
    cudaMalloc(&devicematrix2, matrix_size);
    cudaMalloc(&devicematrix3, matrix_size);
    cudaMemcpy(devicematrix1, hostmatrix1, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(devicematrix2, hostmatrix2, matrix_size, cudaMemcpyHostToDevice);
    GRIDX = (int)ceil((float)m1/BLOCKSIZE);
    GRIDY = (int)ceil((float)n2/BLOCKSIZE);
    dim3 dimGrid(GRIDX, GRIDY, 1);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    MatrixMultiplyI <<< dimGrid, dimBlock >>> (devicematrix1, devicematrix2, devicematrix3, m1, n1, m2, n2);
    cudaMemcpy(hostmatrix3, devicematrix3, matrix_size, cudaMemcpyDeviceToHost);
    write_imatrix(argv[3], &m1, &n1, &hostmatrix3);
    cudaFree(devicematrix1);
    cudaFree(devicematrix2);
    cudaFree(devicematrix3);
    free(hostmatrix1);
    free(hostmatrix2);
    free(hostmatrix3);
  }

  return 0;
}
