#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#include "task_generator.h"
#include "data_types.hpp"
#include "parser.hpp"
#include "gpu_kernels.h"
#include "cpu_kernels.hpp"

#define LINEITEM_PATH "data/lineitem.tbl"

void TaskGenerator::generate()
{
  std::cout << "Generate" << std::endl;
}

void TaskGenerator::run(int r, bool overwrite_file)
{
  LineItem lineitem;
  Parser p;
  p.parse(LINEITEM_PATH, lineitem, overwrite_file);

  std::cout << "Starting program" << std::endl;
  double* l_quantity;
  double* l_extendedprice;
  double* l_discount;
  int* l_shipdate;
  int N = *lineitem.size;

  // Allocate Unified Memory – accessible from CPU or GPU
  std::cout << "Allocating Memory" << std::endl;
  cudaMallocManaged(&l_quantity, N*sizeof(double));
  cudaMallocManaged(&l_extendedprice, N*sizeof(double));
  cudaMallocManaged(&l_discount, N*sizeof(double));
  cudaMallocManaged(&l_shipdate, N*sizeof(int));
  
  std::cout << "Initializing values" << std::endl;
  for (int i = 0; i < N; i++) {
    l_quantity[i] = lineitem.l_quantity[i];
    l_extendedprice[i] = lineitem.l_extendedprice[i];
    l_discount[i] = lineitem.l_discount[i];
    l_shipdate[i] = lineitem.l_shipdate[i];
  }

  int N_cpu = N*r;
  int N_gpu = N*(1-r);
  int blockSize = 256;
  int numBlocks = (N_gpu + blockSize - 1) / blockSize;
  std::cout << "cpu:gpu ratio: " << r << ":" << (1-r) << std::endl;

  auto start = std::chrono::steady_clock::now(); 

  std::cout << "Running kernels" << std::endl;
  check_cpu(N_cpu, l_quantity, l_shipdate, l_discount);
  check<<<numBlocks, blockSize>>>(N_gpu, l_quantity+N_cpu, l_shipdate+N_cpu, l_discount+N_cpu);
  cudaDeviceSynchronize();

  multiply_cpu(N_cpu, l_quantity, l_extendedprice, l_discount);
  multiply<<<numBlocks, blockSize>>>(N_gpu, l_quantity+N_cpu, l_extendedprice+N_cpu, l_discount+N_cpu);
  cudaDeviceSynchronize();

  auto total = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Total time: " << total << " ms" << std::endl;

  // Read out 'query result'
  int amount = 0;
  for (int i = 0; i < N; i++) if (l_extendedprice[i]) amount++;
  std::cout << "Query hit amount: " << amount << std::endl;
  std::cout << "Total tuples: " << N << std::endl;

  // Free memory
  cudaFree(l_discount);
  cudaFree(l_extendedprice);
  cudaFree(l_shipdate);
  cudaFree(l_discount);
}
