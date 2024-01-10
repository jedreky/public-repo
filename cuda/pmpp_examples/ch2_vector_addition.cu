#include <iostream>
#include <math.h>
#include <chrono>
#include <ctime>

using namespace std;

// function to add the elements of two arrays
__global__ void vecAddKernel(float * A, float * B, float * C, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx == 0) {
    printf("blockDim: %d\n", blockDim.x);
    printf("gridDim: %d\n", gridDim.x);
  }

  if (idx < n) {
    C[idx] = A[idx] + B[idx];
  }
}

void performAddition(float * A_h, float * B_h, float * C_h, int N, int size) {
  int blockSize = 256;
  int numBlocks = N / blockSize;

  float * A_d, * B_d, * C_d;
  // the casting is not actually necessary
  cudaMalloc((void**)&A_d, size);
  cudaMalloc((void**)&B_d, size);
  cudaMalloc((void**)&C_d, size);

  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

  cout << "Number of blocks:" << numBlocks << endl;

  vecAddKernel<<<numBlocks, blockSize>>>(A_d, B_d, C_d, N);
  cudaDeviceSynchronize();
  cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

float * getRandomVector(int n) {
  int size = n * sizeof(float);
  float * vec = (float *) malloc(size);

  for (int j=0; j < n; j++) {
    vec[j] = rand();
  }
  return vec;
}

void errorHandling() {
  int N = 1 << 28;
  int size = N * sizeof(float);

  int reps = 18;
  float * vec[reps];
  float * vec_d[reps];
  
  printf("Trying to allocate %d bytes\n", size);

  for (int j=0; j < reps; j++) {
    printf("Attempt %d\n", j);
    cudaError_t result = cudaMalloc(&vec_d[j], size);
    vec[j] = getRandomVector(N);

    cudaMemcpy(vec_d[j], vec[j], size, cudaMemcpyHostToDevice);

    // if (result != cudaSuccess) {
    //   printf("%s in %s at line %d\n", cudaGetErrorString(result), __FILE__, __LINE__);
    //   exit(EXIT_FAILURE);
    // } else {
    //   printf("All good!\n");
    // }
  }
}

int main(void)
{

  // errorHandling();

  int N = 1<<20; // 1M elements
  int size = N * sizeof(float);
  cout << "Number: " << N << endl;

  float * A = (float *) malloc(size);
  float * B = (float *) malloc(size);
  float * C = (float *) malloc(size);

  for (int k = 0; k < 1; k ++)
  {
    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
      A[i] = 1.0f;
      B[i] = 2.0f;
    }

    auto start = std::chrono::system_clock::now();

    performAddition(A, B, C, N, size);

    auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    // // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
      maxError = fmax(maxError, fabs(C[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;
  }

  // Free memory
  // cudaFree(x);
  // cudaFree(y);

  return 0;
}