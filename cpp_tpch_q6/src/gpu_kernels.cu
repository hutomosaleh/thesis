#include "gpu_kernels.h"

// Kernel function to multiply the elements of two arrays
__global__ void multiply(int n, double* l_quantity, double* l_extendedprice, double* l_discount)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    // If WHERE is true, multiply l_extendedprice & l_discount
    l_extendedprice[i] = (l_quantity[i]) ? l_extendedprice[i]*l_discount[i] : 0;
  }
}

// Kernel function to check condition
__global__ void check(int n, double* l_quantity, int* l_shipdate, double* l_extendedprice, double* l_discount)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i+= stride) {
    // Check if WHERE is true, put result in l_quantity
    bool condition = (l_quantity[i]>50 && l_shipdate[i]>50 && l_extendedprice[i]>50 && l_discount[i]>50); // Mock condition
    l_quantity[i] = condition ? 1 : 0;
  }
}

