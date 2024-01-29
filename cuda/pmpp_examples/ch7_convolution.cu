#include <iostream>


#define FILTER_RADIUS 1
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)
#define FILTER_SIZE (FILTER_WIDTH * FILTER_WIDTH)

#define INNER_TILE_WIDTH 3
#define OUTER_TILE_WIDTH (INNER_TILE_WIDTH + 2 * FILTER_RADIUS)

__constant__ float F_constant[FILTER_SIZE];


char * get_main_folder() {
  char * main_folder = (char *) malloc( sizeof(char) * 100 );
  strcpy(main_folder, "/home/legion/open-source/public-repo/cuda/pmpp_examples/");
  return main_folder;
}


float * read_vector_from_file(int size, const char * filename) {
  char * full_path = get_main_folder();
  strcat(full_path, filename);

  FILE * fptr;
  fptr = fopen(full_path, "rb");

  float * v = (float *) malloc( size * sizeof(float) );

  fread((void*)(v), sizeof(v), size, fptr);
  fclose(fptr);
  return v;
}


__global__ void conv_basic_kernel(float * X, float * F, float * Y, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    float val = 0;

    int inRow, inCol;

    for (int fCol = 0; fCol < FILTER_WIDTH; fCol++) {
      inCol = col - FILTER_RADIUS + fCol;

      for (int fRow = 0; fRow < FILTER_WIDTH; fRow++) {      
        inRow = row - FILTER_RADIUS + fRow;
        
        if (inRow >=0 && inRow < height && inCol >= 0 && inCol < width) {
          val += F[fRow * FILTER_WIDTH + fCol] * X[inRow * width + inCol];
        }
      }
    }
    Y[row * width + col] = val;
  }
}


__global__ void conv_basic_kernel_constant(float * X, float * Y, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    float val = 0;

    int inRow, inCol;

    for (int fCol = 0; fCol < FILTER_WIDTH; fCol++) {
      inCol = col - FILTER_RADIUS + fCol;

      for (int fRow = 0; fRow < FILTER_WIDTH; fRow++) {      
        inRow = row - FILTER_RADIUS + fRow;
        
        if (inRow >=0 && inRow < height && inCol >= 0 && inCol < width) {
          val += F_constant[fRow * FILTER_WIDTH + fCol] * X[inRow * width + inCol];
        }
      }
    }
    Y[row * width + col] = val;
  }
}


__global__ void conv_kernel_tiled(float * X, float * Y, int width, int height, int nums_to_load) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = row * width + col;

  __shared__ float Xtile[OUTER_TILE_WIDTH][OUTER_TILE_WIDTH];

  if (col >= 0 && col < width && row >= 0 && row < height) {
    for (int k=0; k < nums_to_load; k++) {

      int current_idx = idx * nums_to_load + k;
      // printf("current idx %d\n", current_idx);
      int current_col = current_idx % OUTER_TILE_WIDTH;
      int current_row = current_idx / OUTER_TILE_WIDTH;

      if (current_col - FILTER_RADIUS >= 0 && current_col - FILTER_RADIUS < width && current_row - FILTER_RADIUS >= 0 && current_row - FILTER_RADIUS < height) {
        // printf("%d, %d %d: %d %d loaded\n", current_idx, current_col, current_row, current_col - FILTER_RADIUS, current_row - FILTER_RADIUS);
        Xtile[current_row][current_col] = X[(current_row - FILTER_RADIUS) * width + current_col - FILTER_RADIUS];
      }
    }

    __syncthreads();

    float val = 0;

    int inRow, inCol;

    for (int fCol = 0; fCol < FILTER_WIDTH; fCol++) {
      inCol = col - FILTER_RADIUS + fCol;

      for (int fRow = 0; fRow <= FILTER_WIDTH; fRow++) {      
        inRow = row - FILTER_RADIUS + fRow;
        
        if (inRow >=0 && inRow < height && inCol >= 0 && inCol < width) {
          val += F_constant[fRow * FILTER_WIDTH + fCol] * Xtile[inRow + FILTER_RADIUS][inCol + FILTER_RADIUS];
        }
      }
    }
    Y[row * width + col] = val;
    __syncthreads();
  }
}


int main() {
  char * filename_in = (char *) "ch7_input.dat";
  int m = 3;
  int n = 3;
  int size = m * n;

  float * X_h = read_vector_from_file(size, filename_in);
  float * F_h = (float *) malloc(FILTER_SIZE * sizeof(float));

  for (int j=0; j < FILTER_SIZE; j++) {
    F_h[j] = j;
  }

  dim3 gridSize(ceil((float)m/INNER_TILE_WIDTH), ceil((float)n/INNER_TILE_WIDTH), 1);
  dim3 blockSize(INNER_TILE_WIDTH, INNER_TILE_WIDTH, 1);

  float * X_d, * Y_d, * F_d;
  cudaMalloc((void**) &X_d, size * sizeof(float));
  cudaMalloc((void**) &Y_d, size * sizeof(float));
  float * Y_h = (float *) malloc(size * sizeof(float));

  cudaMemcpy(X_d, X_h, size * sizeof(float), cudaMemcpyHostToDevice);

  int mode = 0;
  clock_t time_start, time_end;

  if (mode == 0) {
    cudaMalloc((void**) &F_d, FILTER_SIZE * sizeof(float));
    cudaMemcpy(F_d, F_h, FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    time_start = clock();
    conv_basic_kernel<<<gridSize, blockSize>>>(X_d, F_d, Y_d, m, n);
    cudaDeviceSynchronize();

    time_end = clock();
  } else {
    cudaMemcpyToSymbol(F_constant, F_h, FILTER_SIZE * sizeof(float));

    if (mode == 1) {
      time_start = clock();
      conv_basic_kernel_constant<<<gridSize, blockSize>>>(X_d, Y_d, m, n);
      cudaDeviceSynchronize();

      time_end = clock();
    } else if (mode == 2) {
      int nums_to_load = ceil((float)(OUTER_TILE_WIDTH * OUTER_TILE_WIDTH) / (INNER_TILE_WIDTH * INNER_TILE_WIDTH));

      printf("Nums to load: %d\n", nums_to_load);

      time_start = clock();
      conv_kernel_tiled<<<gridSize, blockSize>>>(X_d, Y_d, m, n, nums_to_load);
      cudaDeviceSynchronize();

      time_end = clock();
    }

  }

  printf("Elapsed: %.5f seconds\n\n", (double)(time_end - time_start) / CLOCKS_PER_SEC);
  cudaMemcpy(Y_h, Y_d, size * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Input:\n");

  for (int j=0; j < min(m * n, 9); j++) {
    printf("%.2f ", X_h[j]);
  }

  printf("\n\nOutput:\n");

  for (int j=0; j < min(m * n, 9); j++) {
    printf("%.2f ", Y_h[j]);
  }

  printf("\n\n");

  cudaFree(X_d);

  if (mode == 0) {
    cudaFree(Y_d);
  }

  cudaFree(F_d);
}