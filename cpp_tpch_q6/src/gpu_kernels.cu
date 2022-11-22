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
__global__ void check(int n, double* l_quantity, int* l_shipdate, double* l_discount)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i+= stride) {
    // Check if WHERE is true, put result in l_quantity
    bool valid_date = (l_shipdate[i] >= 727841 && l_shipdate[i] <= 728206);
    bool valid_quantity = (l_quantity[i] < 24.0);
    bool valid_discount = (l_discount[i] > 0.05 && l_discount[i] < 0.07);
    l_quantity[i] = (valid_date && valid_quantity && valid_discount) ? 1 : 0;
  }
}

