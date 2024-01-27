#include <chrono>
#include <ctime>
#include <iostream>
#include <math.h>

using namespace std;

const int CHANNELS = 3;


char * get_main_folder() {
  char * main_folder = (char *) malloc( sizeof(char) * 100 );
  strcpy(main_folder, "/home/legion/open-source/public-repo/cuda/pmpp_examples/");
  return main_folder;
}


unsigned char * read_vector_from_file(int size, const char * filename) {
  char * full_path = get_main_folder();
  strcat(full_path, filename);

  FILE * fptr;
  fptr = fopen(full_path, "rb");

  unsigned char * v = (unsigned char *) malloc( size * CHANNELS * sizeof(unsigned char) );

  fread((void*)(v), sizeof(v), size, fptr);
  fclose(fptr);
  return v;
}


void write_vector_to_file(unsigned char * img_out, int size, const char * filename) {
  char * full_path = get_main_folder();
  strcat(full_path, filename);

  FILE *fptr;
  fptr = fopen(full_path, "wb");

  fwrite(img_out, sizeof(unsigned char), size, fptr);
}


__global__ void conversionKernel(unsigned char * img_in, unsigned char * img_out, int m, int n)
{
  int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (col_idx < m && row_idx < n) {
    int grayPos = row_idx * m + col_idx;
    int rgbPos = grayPos * CHANNELS;
    unsigned char r = img_in[rgbPos];
    unsigned char g = img_in[rgbPos + 1];
    unsigned char b = img_in[rgbPos + 2];
    float float_result = 0.21 * r + 0.71 * g + 0.07 * b;
    img_out[grayPos] = (unsigned char) round(float_result);
  }
}


__global__ void blurKernel(unsigned char * img_in, unsigned char * img_out, int m, int n)
{
  int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned short int avg;
  int pos;

  if (col_idx < m && row_idx < n) {
    int cols[3] = {col_idx - 1, col_idx, col_idx + 1};
    int rows[3] = {row_idx - 1, row_idx, row_idx + 1};

    if (col_idx == 0) {
      cols[0] = 0;
    } else if (col_idx == m - 1) {
      cols[2] = m - 1;
    }

    if (row_idx == 0) {
      rows[0] = 0;
    } else if (row_idx == n - 1) {
      rows[2] = n - 1;
    }

    for (int j=0; j < 3; j++) {
      avg = 0;

      for (int k=0; k < 3; k++) {
        for (int l=0; l < 3; l++) {
          pos = (rows[k] * m + cols[l]) * CHANNELS;
          
          avg += (unsigned short int) img_in[pos + j];
        }
      }

      img_out[(row_idx * m + col_idx) * CHANNELS + j] = (unsigned char) (avg/9);
    }
  }
}


unsigned char * process(unsigned char * img_in_h, int m, int n, bool to_grayscale) {
  dim3 gridSize(ceil(m/16.0), ceil(n/16.0), 1);
  dim3 blockSize(16, 16, 1);

  int size = m * n;
  int input_size = CHANNELS * size;

  int output_size;

  if (to_grayscale) {
    output_size = size;
  } else {
    output_size = input_size;
  }

  unsigned char * img_in_d, * img_out_d;
  cudaMalloc((void**) &img_in_d, input_size);

  cudaMalloc((void**) &img_out_d, output_size);
  unsigned char * img_out_h = (unsigned char *) malloc(output_size);

  cudaMemcpy(img_in_d, img_in_h, input_size, cudaMemcpyHostToDevice);

  if (to_grayscale) {
    conversionKernel<<<gridSize, blockSize>>>(img_in_d, img_out_d, m, n);
  } else {
    blurKernel<<<gridSize, blockSize>>>(img_in_d, img_out_d, m, n);
  }
  
  cudaDeviceSynchronize();
  cudaMemcpy(img_out_h, img_out_d, output_size, cudaMemcpyDeviceToHost);

  cudaFree(img_in_d);
  cudaFree(img_out_d);

  return img_out_h;
}


int main(void)
{
  char * filename_in = (char *) "ch3_img_in.dat";
  int m = 960;
  int n = 640;

  bool to_grayscale = true;
  // to_grayscale = false;

  unsigned char * img = read_vector_from_file(m * n, filename_in);

  unsigned char * img_out;
  int output_size;
  char * filename_out;

  auto start = std::chrono::system_clock::now();

  img_out = process(img, m, n, to_grayscale);

  auto end = std::chrono::system_clock::now();

  std::chrono::duration<float> elapsed_seconds = end - start;

  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

  if (to_grayscale) {
    output_size = m * n;
    filename_out = (char *) "ch3_img_convert_out.dat";
  } else {
    output_size = m * n * CHANNELS;
    filename_out = (char *) "ch3_img_blur_out.dat";
  }

  write_vector_to_file(img_out, output_size, filename_out);

  return 0;
}