#include <iostream>

// based on https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html


char * format_memory(int bytes) {
  char * ret = (char *) malloc(50 * sizeof(char));

  if (bytes >= 1024*1024 && bytes % 1024*1024 == 0) {
    sprintf(ret, "[MB]: %d", bytes / (1024*1024));
  } else if (bytes >= 1024 and bytes % 1024 == 0) {
    sprintf(ret, "[KB]: %d", bytes / 1024);
  } else {
    sprintf(ret, "[B]: %d", bytes);
  }

  return ret;
}


int main(void)
{
  int devCount;
  cudaGetDeviceCount(&devCount);
  printf("Found %d CUDA device(s)\n", devCount);

  cudaDeviceProp devProp;
  // assuming there is only 1 device
  cudaGetDeviceProperties(&devProp, 0);

  printf("Name: %s\n", devProp.name);
  printf("Number of SMs: %d\n", devProp.multiProcessorCount);
  printf("Clock rate [kHz]: %d\n\n", devProp.clockRate);

  printf("Max number of threads per block: %d\n", devProp.maxThreadsPerBlock);
  printf("Max number of threads in x, y, z dimensions: %d, %d, %d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
  printf("Max number of blocks in x, y, z dimensions: %d, %d, %d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
  printf("Warp size: %d\n\n", devProp.warpSize);

  printf("Number of 32-bit registers available per block: %d\n", devProp.regsPerBlock);
  printf("Number of 32-bit registers available per SM: %d\n", devProp.regsPerMultiprocessor);
  printf("Shared memory available to each SM %s\n", format_memory(devProp.sharedMemPerBlock));
  printf("Total global memory %s\n", format_memory(devProp.totalGlobalMem));

  return 0;
}
