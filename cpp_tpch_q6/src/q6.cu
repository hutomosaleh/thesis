#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#include "data_types.hpp"
#include "parser.hpp"
#include "gpu_kernels.h"
#include "defs.hpp"

void check_cpu(int n, double* l_quantity, int* l_shipdate, double* l_discount)
{
  for (int i = 0; i < n; i++) {
    bool valid_date = (l_shipdate[i] >= DATE_BOTTOM_LIMIT && l_shipdate[i] <= DATE_UPPER_LIMIT);
    bool valid_quantity = (l_quantity[i] < QUANTITY_LIMIT);
    bool valid_discount = (l_discount[i] >= DISCOUNT_BOTTOM_LIMIT && l_discount[i] < DISCOUNT_UPPER_LIMIT);
    l_quantity[i] = (valid_date && valid_quantity && valid_discount) ? 1 : 0;
  }
}

void multiply_cpu(int n, double* l_quantity, double* l_extendedprice, double* l_discount)
{
  for (int i = 0; i < n; i++) {
    l_extendedprice[i] = (l_quantity[i]) ? l_extendedprice[i]*l_discount[i] : 0;
  }
}

int main(int argc, char** argv)
{
  float r = 1.0;
  bool overwrite_file = false;
  if (argc > 1) {
    r = atof(argv[1]);
    std::cout << "Ratio: " << r << std::endl;
    if (argc > 2)
    {
      std::string str(argv[2]);
      if (str == "overwrite") overwrite_file = true;
    }
  } else { std::cout << "Ratio set to default: " << r << std::endl; }
  
  LineItem lineitem;
  Parser p;
  p.parse(LINEITEM_PATH, lineitem, overwrite_file);

  auto start1 = std::chrono::steady_clock::now(); 
  std::cout << "Starting program" << std::endl;
  int N = *lineitem.size;

  int N_cpu = N*(1-r);
  int N_gpu = N*r;
  int blockSize = 256;
  int numBlocks = (N_gpu + blockSize - 1) / blockSize;
  std::cout << "cpu:gpu ratio: " << N_cpu << ":" << N_gpu << std::endl;

  // Allocate host memory
  std::cout << "Allocating Memory" << std::endl;
  double* l_quantity;
  double* l_extendedprice;
  double* l_discount;
  int* l_shipdate;
  cudaMallocHost(&l_quantity, N*sizeof(double));
  cudaMallocHost(&l_extendedprice, N*sizeof(double));
  cudaMallocHost(&l_discount, N*sizeof(double));
  cudaMallocHost(&l_shipdate, N*sizeof(int));

  
  // Allocate device memory
  double* q;
  double* e;
  double* d;
  int* s;
  cudaMalloc(&q, N_gpu*sizeof(double));
  cudaMalloc(&e, N_gpu*sizeof(double));
  cudaMalloc(&d, N_gpu*sizeof(double));
  cudaMalloc(&s, N_gpu*sizeof(int));

  std::cout << "Initializing values" << std::endl;
  for (int i = 0; i < N; i++) {
    l_quantity[i] = lineitem.l_quantity[i];
    l_extendedprice[i] = lineitem.l_extendedprice[i];
    l_discount[i] = lineitem.l_discount[i];
    l_shipdate[i] = lineitem.l_shipdate[i];
  }

  cudaMemcpy(q, l_quantity+N_cpu, N_gpu*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(e, l_extendedprice+N_cpu, N_gpu*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d, l_discount+N_cpu, N_gpu*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(s, l_shipdate+N_cpu, N_gpu*sizeof(int), cudaMemcpyHostToDevice);


  auto total1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start1).count();
  auto start2 = std::chrono::steady_clock::now(); 
  std::cout << "Running kernels" << std::endl;
  check_cpu(N_cpu, l_quantity, l_shipdate, l_discount);
  check<<<numBlocks, blockSize>>>(N_gpu, q, s, d);
  cudaDeviceSynchronize();

  auto total2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start2).count();

  auto start3 = std::chrono::steady_clock::now(); 
  multiply_cpu(N_cpu, l_quantity, l_extendedprice, l_discount);
  multiply<<<numBlocks, blockSize>>>(N_gpu, q, e, d);
  cudaDeviceSynchronize();
  auto total3 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start3).count();

  cudaMemcpy(l_quantity+N_cpu, q, N_gpu*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(l_extendedprice+N_cpu, e, N_gpu*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(l_discount+N_cpu, d, N_gpu*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(l_shipdate+N_cpu, s, N_gpu*sizeof(int), cudaMemcpyDeviceToHost);

  auto total4 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start1).count();
  std::cout << "Total time alloc: " << total1 << " ms" << std::endl;
  std::cout << "Total time check: " << total2 << " ms" << std::endl;
  std::cout << "Total time multiply: " << total3 << " ms" << std::endl;
  std::cout << "Total time : " << total4 << " ms" << std::endl;

  // Read out 'query result'
  int amount = 0;
  double result = 0;
  for (int i = 0; i < N; i++)
  {
    if (l_extendedprice[i]) 
    {
      amount++;
      result += l_extendedprice[i];
    }
  }
  std::cout << "Query hit amount: " << amount << std::endl;
  std::cout << "Total tuples: " << N << std::endl;
  std::cout << std::fixed <<  "Result: " << result << std::endl;

  // Free memory
  cudaFree(l_quantity);
  cudaFree(l_extendedprice);
  cudaFree(l_discount);
  cudaFree(l_shipdate);
  cudaFree(q);
  cudaFree(e);
  cudaFree(d);
  cudaFree(s);
  
  return 0;
}
