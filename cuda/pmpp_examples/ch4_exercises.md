5. If we launch a kernel with 32 threads, it will be launched as a single warp meaning all the threads will start at the same time. However, there is no guarantee the processing will take the same time. Therefore, we still have to use `__syncthreads()`.

6. c
