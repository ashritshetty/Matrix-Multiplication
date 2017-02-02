#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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

void multiply_imatrix(int **input1, int **input2, int **output, int *m, int *n, int *p, int *q)
{
	int i, j, k;
	int offset1, offset2, offset3;
	int sum = 0;
	for (i = 0; i < *m; i++)
	{
    for (j = 0; j < *q; j++)
		{
      for (k = 0; k < *p; k++)
			{
				offset1 = i*(*n) + k;
				offset2 = k*(*q) + j;
        sum = sum + *(*input1+offset1) * *(*input2+offset2);
      }
			offset3 = i*(*q) + j;
 			*(*output+offset3) = sum;
      sum = 0.0;
    }
  }
}

void multiply_fmatrix(float **input1, float **input2, float **output, int *m, int *n, int *p, int *q)
{
	int i, j, k;
	int offset1, offset2, offset3;
	float sum = 0.0;
	for (i = 0; i < *m; i++)
	{
    for (j = 0; j < *q; j++)
		{
      for (k = 0; k < *p; k++)
			{
				offset1 = i*(*n) + k;
				offset2 = k*(*q) + j;
        sum = sum + *(*input1+offset1) * *(*input2+offset2);
      }
			offset3 = i*(*q) + j;
 			*(*output+offset3) = sum;
      sum = 0.0;
    }
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
    read_fmatrix(argv[1], &m1, &n1, &hostmatrix1);
  	read_fmatrix(argv[2], &m2, &n2, &hostmatrix2);
    matrix_check(m1, n1, m2, n2);
    matrix_size = m1 * n1;
    hostmatrix3 = (float *)calloc(matrix_size, sizeof(float));
    multiply_fmatrix(&hostmatrix1, &hostmatrix2, &hostmatrix3, &m1, &n1, &m2, &n2);
    write_fmatrix(argv[3], &m1, &n1, &hostmatrix3);
    free(hostmatrix1);
    free(hostmatrix2);
    free(hostmatrix3);
  }

  if (strcmp(argv[4], "int") == 0)
  {
    int *hostmatrix1, *hostmatrix2, *hostmatrix3;
    read_imatrix(argv[1], &m1, &n1, &hostmatrix1);
  	read_imatrix(argv[2], &m2, &n2, &hostmatrix2);
    matrix_check(m1, n1, m2, n2);
    matrix_size = m1 * n1;
    hostmatrix3 = (int *)calloc(matrix_size, sizeof(int));
    multiply_imatrix(&hostmatrix1, &hostmatrix2, &hostmatrix3, &m1, &n1, &m2, &n2);
    write_imatrix(argv[3], &m1, &n1, &hostmatrix3);
    free(hostmatrix1);
    free(hostmatrix2);
    free(hostmatrix3);
  }

  return 0;
}
