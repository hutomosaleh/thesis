#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <random>
#include <sstream>
#include <string>
#include "gpu_kernels.h"

#define QUANTITY 5
#define EXTENDED_PRICE 6
#define DISCOUNT 7
#define SHIPDATE 11
#define NUM_COLUMN 16
#define LINEITEM_PATH "cpp_tpch_q6/data/lineitem.tbl"
#define DELIMITER '|'

void check_cpu(int n, double* l_quantity, int* l_shipdate, double* l_extendedprice, double* l_discount)
{
  for (int i = 0; i < n; i++) {
    bool condition = (l_quantity[i]>50 && l_shipdate[i]>50 && l_extendedprice[i]>50 && l_discount[i]>50); // Mock condition
    l_quantity[i] = condition ? 1 : 0;
  }
}

void multiply_cpu(int n, double* l_quantity, double* l_extendedprice, double* l_discount)
{
  for (int i = 0; i < n; i++) {
    l_extendedprice[i] = (l_quantity[i]) ? l_extendedprice[i]*l_discount[i] : 0;
  }
}

struct LineItem {
  std::vector<double> l_quantity;
  std::vector<double> l_extendedprice;
  std::vector<double> l_discount;
  std::vector<int> l_shipdate;
  int size = 0;
};

int dtoi(std::string str) {
  std::istringstream date(str);
  std::string time;
  int result = 0;
  int count = 0;
  while (getline(date, time, '-')) {
    int multiplier = 1;
    if (count==1) multiplier=365;
    if (count==2) multiplier=30;
    ++count;
    result += stoi(time)*multiplier;
  }
  return result;
}

void parse_lineitem(std::string path, LineItem& record)
{ 
  std::cout << "Parsing lineitem" << std::endl;
  std::fstream buffer(path);
  std::string line;
  while (getline(buffer, line)) {
    std::istringstream row(line);
    std::string field;
    int column;
    while (getline(row, field, DELIMITER)) {
      if (column==QUANTITY) {
        record.l_quantity.push_back(std::stod(field));
      } else if (column==EXTENDED_PRICE) {
        record.l_extendedprice.push_back(std::stod(field));
      } else if (column==DISCOUNT) {
        record.l_discount.push_back(std::stod(field));
      } else if (column==SHIPDATE) {
        record.l_shipdate.push_back(dtoi(field));
      }
      ++column;
      if (column==NUM_COLUMN) {
        column = 0;
        continue;
      }
    }
    ++record.size;
  }
}

int main(int argc, char** argv)
{
  float r = 1.0;
  LineItem lineitem;
  parse_lineitem(LINEITEM_PATH, lineitem);
  if (argc > 1) {
    r = atof(argv[1]);
    std::cout << "Ratio: " << r << std::endl;
  } else { std::cout << "Ratio set to default: " << r << std::endl; }

  std::cout << "Starting program" << std::endl;
  double* l_quantity = &lineitem.l_quantity[0];
  double* l_extendedprice = &lineitem.l_extendedprice[0];
  double* l_discount = &lineitem.l_discount[0];
  int* l_shipdate = &lineitem.l_shipdate[0];
  int N = lineitem.size;

  // Allocate Unified Memory – accessible from CPU or GPU
  std::cout << "Allocating Memory" << std::endl;
  cudaMallocManaged(&l_quantity, N*sizeof(double));
  cudaMallocManaged(&l_extendedprice, N*sizeof(double));
  cudaMallocManaged(&l_discount, N*sizeof(double));
  cudaMallocManaged(&l_shipdate, N*sizeof(int));

  int N_cpu = N*r;
  int N_gpu = N*(1-r);
  int blockSize = 256;
  int numBlocks = (N_gpu + blockSize - 1) / blockSize;
  std::cout << "cpu:gpu ratio: " << r << ":" << (1-r) << std::endl;

  auto start = std::chrono::steady_clock::now(); 

  std::cout << "Running kernels" << std::endl;
  check_cpu(N_cpu, l_quantity, l_shipdate, l_extendedprice, l_discount);
  check<<<numBlocks, blockSize>>>(N_gpu, l_quantity+N_cpu, l_shipdate+N_cpu, l_extendedprice+N_cpu, l_discount+N_cpu);
  cudaDeviceSynchronize();

  multiply_cpu(N_cpu, l_quantity, l_extendedprice, l_discount);
  multiply<<<numBlocks, blockSize>>>(N_gpu, l_quantity+N_cpu, l_extendedprice+N_cpu, l_discount+N_cpu);
  cudaDeviceSynchronize();

  auto total = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << "Total time: " << total << " ms" << std::endl;

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
