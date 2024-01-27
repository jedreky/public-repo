#include <iostream>


#define FILTER_RADIUS 63


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


__global__ void conv_kernel_tiled(float * N, float * F, float * P, int r, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x - r;
  int row = blockIdx.y * blockDim.y + threadIdx.y - r;

  __shared__ float * N_loaded[r][r];

  if (col >= 0 && col <= width && row >= 0 && row <= height) {
    N_loaded[col][row] = N[col * r + row];
  }
  __syncthreads();

  if (col <= width && row <= height) {
    float Pval = 0;

    int inRow, inCol;

    for (int fCol = 0; fCol <= 2 * r; fCol++) {
      inCol = col - r + fCol;

      for (int fRow = 0; fRow <= 2 * r; fRow++) {      
        inRow = row - r + fRow;
        
        if (inRow >=0 && inRow < height && inCol >= 0 && inCol < width) {
          Pval += F[fRow * r + fCol] * N[inRow * width + inCol];
        }
      }
    }
    P[row * width + col] = Pval;
    __syncthreads();
  }
}


int main() {
  char * filename_in = (char *) "ch7_input.dat";
  int m = 10000;
  int n = 10000;
  int size = m * n;

  int filter_size = (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1);

  float * img_in_h = read_vector_from_file(size, filename_in);
  float * F_h = (float *) malloc(filter_size * sizeof(float));

  for (int j=0; j < filter_size; j++) {
    F_h[j] = j;
  }

  dim3 gridSize(ceil(m/16.0), ceil(n/16.0), 1);
  dim3 blockSize(16, 16, 1);

  float * img_in_d, * img_out_d, * F_d;
  cudaMalloc((void**) &img_in_d, size * sizeof(float));
  cudaMalloc((void**) &img_out_d, size * sizeof(float));
  cudaMalloc((void**) &F_d, filter_size * sizeof(float));
  float * img_out_h = (float *) malloc(size * sizeof(float));

  cudaMemcpy(img_in_d, img_in_h, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(F_d, F_h, filter_size * sizeof(float), cudaMemcpyHostToDevice);

  clock_t time_start = clock();
  conv_basic_kernel<<<gridSize, blockSize>>>(img_in_d, F_d, img_out_d, FILTER_RADIUS, m, n);
  cudaDeviceSynchronize();

  clock_t time_end = clock();

  printf("Elapsed: %.5f seconds\n\n", (double)(time_end - time_start) / CLOCKS_PER_SEC);
  cudaMemcpy(img_out_h, img_out_d, size * sizeof(float), cudaMemcpyDeviceToHost);

  // for (int j=0; j<m * n; j++) {
    // printf("%f.2f ", img_in_h[j]);
    // printf("%.2f ", img_out_h[j]);
  // }

  cudaFree(img_in_d);
  cudaFree(img_out_d);
  cudaFree(F_d);
}