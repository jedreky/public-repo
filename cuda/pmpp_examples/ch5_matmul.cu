#include <iostream>
#include <assert.h>
#include <math.h>
#include <time.h>


// basic matrix multiplication
__global__ void matMulKernel(float * A, float * B, float * C, int left_dim, int mid_dim, int right_dim)
{
  int col_idx, row_idx;
  bool correct_access_pattern = true;

  if (correct_access_pattern) {
    col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  } else {
    row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  }

  if (col_idx < right_dim && row_idx < left_dim) {
    int target_idx = row_idx * right_dim + col_idx;
    C[target_idx] = 0;

    for (int j = 0; j < mid_dim; j++) {
      C[target_idx] += A[row_idx * mid_dim + j] * B[j * right_dim + col_idx];
    }
  } else {
    printf("Skipped kernel call!\n");
  }
}


// tiled matrix multiplication, collaborative data loading
#define TILE_WIDTH 32
__global__ void matMulKernelTiled(float * A, float * B, float * C, int left_dim, int mid_dim, int right_dim)
{
  assert(blockDim.x == TILE_WIDTH);
  assert(blockDim.y == TILE_WIDTH);

  __shared__ float ATile[TILE_WIDTH][TILE_WIDTH];
  __shared__ float BTile[TILE_WIDTH][TILE_WIDTH];

  // int warpid;
  // asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  // printf("Warp %d, bx %d, by %d, tx %d, ty %d\n", warpid, bx, by, tx, ty);

  int col_idx = bx * blockDim.x + tx;
  int row_idx = by * blockDim.y + ty;

  float val = 0;

  for (int j = 0; j < mid_dim / TILE_WIDTH; j++)
  {
    ATile[ty][tx] = A[row_idx * mid_dim + j * TILE_WIDTH + tx];
    BTile[ty][tx] = B[(j * TILE_WIDTH + ty) * right_dim + col_idx];
    // the following line shows how B should be loaded if it was stored in a column-major format
    // BTile[ty][tx] = B[col_idx * right_dim + (j * TILE_WIDTH + ty)];
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++) {
      val += ATile[ty][k] * BTile[k][tx];
    }
    __syncthreads();
  }
  C[row_idx * right_dim + col_idx] = val;
}


void matMul(float * A_h, float * B_h, float * C_h, int left_dim, int mid_dim, int right_dim) {
  dim3 blockSize(32, 32);
  dim3 gridSize((right_dim - 1) / 32 + 1, (left_dim - 1) / 32 + 1);
  printf("Required grid: %d x %d\n\n", gridSize.x, gridSize.y);

  int sizeA = left_dim * mid_dim * sizeof(float);
  int sizeB = mid_dim * right_dim * sizeof(float);
  int sizeC = left_dim * right_dim * sizeof(float);

  float * A_d, * B_d, * C_d;
  // the casting is not actually necessary
  cudaMalloc((void**)&A_d, sizeA);
  cudaMalloc((void**)&B_d, sizeB);
  cudaMalloc((void**)&C_d, sizeC);

  cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice);

  time_t start, end;

  time(&start);
  matMulKernel<<<gridSize, blockSize>>>(A_d, B_d, C_d, left_dim, mid_dim, right_dim);
  // matMulKernelTiled<<<gridSize, blockSize>>>(A_d, B_d, C_d, left_dim, mid_dim, right_dim);
  cudaDeviceSynchronize();
  time(&end);
  int time_elapsed = end - start;
  printf("Time elapsed: %d secs\n", time_elapsed);

  cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}


int main(void)
{

  int dim = 1024 * 32;
  dim = 1024 * 16;
  int left_dim = dim;
  int mid_dim = dim;
  int right_dim = dim;

  float * A = (float *) malloc(left_dim * mid_dim * sizeof(float));
  float * B = (float *) malloc(mid_dim * right_dim * sizeof(float));
  float * C = (float *) malloc(left_dim * right_dim * sizeof(float));

  for (int j = 0; j < left_dim * mid_dim; j++) {
    A[j] = 0.2 * (-5 + (j % 10));
  }

  for (int j = 0; j < mid_dim * right_dim; j++) {
    B[j] = 0.1 * (0.1 + (j % 8));
  }

  matMul(A, B, C, left_dim, mid_dim, right_dim);

  printf("Output:\n");
  for (int j = 0; j < 3; j++) {
    printf("%f\n", C[j]);
  }
  printf("...\n");
  for (int j = left_dim * right_dim - 3; j < left_dim * right_dim; j++) {
    printf("%f\n", C[j]);
  }

  return 0;
}
