#include <iostream>
#include <math.h>
#include <random>

// Kernel function to multiply the elements of two arrays
__global__
void multiply(int n, int* a, int* x, int* y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    // If WHERE is true, multiply l_extendedprice & l_discount
    x[i] = (a[i]) ? x[i]*y[i] : 0;
  }
}

// Kernel function to check condition
__global__
void check(int n, int* a, int* b, int* c, int* d) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i+= stride) {
    // Check if WHERE is true, put result in l_quantity
    bool condition = (a[i]>50 && b[i]>50 && c[i]>50 && d[i]>50); // Mock condition
    a[i] = condition ? 1 : 0;
  }
}

// CPU function
void check_cpu(int n, int* a, int* b, int* c, int* d) {
  for (int i = 0; i < n; i++) {
    bool condition = (a[i]>50 && b[i]>50 && c[i]>50 && d[i]>50); // Mock condition
    a[i] = condition ? 1 : 0;
  }
}

int main(void)
{
  std::cout << "Starting program" << std::endl;
  int N = 1<<20;
  int* l_shipdate;
  int* l_quantity;
  int* l_extendedprice;
  int* l_discount;

  // Allocate Unified Memory – accessible from CPU or GPU
  std::cout << "Allocating Memory" << std::endl;
  cudaMallocManaged(&l_extendedprice, N*sizeof(int));
  cudaMallocManaged(&l_discount, N*sizeof(int));
  cudaMallocManaged(&l_quantity, N*sizeof(int));
  cudaMallocManaged(&l_shipdate, N*sizeof(int));

  // initialize rng
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> generateRandomInt(1, 100);

  // initialize x and y arrays on the host
  std::cout << "Initializing values" << std::endl;
  for (int i = 0; i < N; i++) {
    l_extendedprice[i] = generateRandomInt(rng);
    l_discount[i] = generateRandomInt(rng);
    l_shipdate[i] = generateRandomInt(rng);
    l_quantity[i] = generateRandomInt(rng);
  }

  // Run kernel on 1M elements on the GPU
  std::cout << "Running kernels" << std::endl;
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  check_cpu(N, l_quantity, l_shipdate, l_extendedprice, l_discount);

  //check<<<numBlocks, blockSize>>>(N, l_quantity, l_shipdate, l_extendedprice, l_discount);
  // cudaDeviceSynchronize();  // Is this necessary?

  multiply<<<numBlocks, blockSize>>>(N, l_quantity, l_extendedprice, l_discount);
  cudaDeviceSynchronize();

  // Read out 'query result'
  int amount = 0;
  for (int i = 0; i < N; i++) if (l_extendedprice[i]) amount++;
  std::cout << "Amount: " << amount << std::endl;
  std::cout << "N: " << N << std::endl;

  // Free memory
  cudaFree(l_discount);
  cudaFree(l_extendedprice);
  cudaFree(l_shipdate);
  cudaFree(l_discount);
   
  return 0;
}
