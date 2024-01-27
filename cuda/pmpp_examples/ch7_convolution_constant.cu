#include <iostream>


#define FILTER_RADIUS 63
__constant__ float F_d[(2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1)];


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


__global__ void conv_basic_kernel(float * N, float * P, int r, int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col <= width && row <= height) {
    float Pval = 0;

    int inRow, inCol;

    for (int fCol = 0; fCol <= 2 * r; fCol++) {
      inCol = col - r + fCol;

      for (int fRow = 0; fRow <= 2 * r; fRow++) {      
        inRow = row - r + fRow;
        
        if (inRow >=0 && inRow < height && inCol >= 0 && inCol < width) {
          Pval += F_d[fRow * r + fCol] * N[inRow * width + inCol];
        }
      }
    }
    P[row * width + col] = Pval;
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
  cudaMemcpyToSymbol(F_d, F_h, filter_size * sizeof(float));

  clock_t time_start = clock();
  conv_basic_kernel<<<gridSize, blockSize>>>(img_in_d, img_out_d, FILTER_RADIUS, m, n);
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