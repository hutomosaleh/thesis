#include "gpu_kernels.h"

// Kernel function to multiply the elements of two arrays
__global__ void multiply(int n, int* a, int* x, int* y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    // If WHERE is true, multiply l_extendedprice & l_discount
    x[i] = (a[i]) ? x[i]*y[i] : 0;
  }
}

// Kernel function to check condition
__global__ void check(int n, int* a, int* b, int* c, int* d) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i+= stride) {
    // Check if WHERE is true, put result in l_quantity
    bool condition = (a[i]>50 && b[i]>50 && c[i]>50 && d[i]>50); // Mock condition
    a[i] = condition ? 1 : 0;
  }
}

