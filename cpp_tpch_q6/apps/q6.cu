#include <iostream>
#include <iterator>
#include <istream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <random>
#include <sstream>
#include <string>
#include "gpu_kernels.h"

#define QUANTITY 4
#define EXTENDED_PRICE 5
#define DISCOUNT 6
#define SHIPDATE 10
#define NUM_COLUMN 16
#define LINEITEM_PATH "cpp_tpch_q6/data/lineitem.tbl"
#define DELIMITER '|'

void check_cpu(int n, double* l_quantity, int* l_shipdate, double* l_discount)
{
  for (int i = 0; i < n; i++) {
    bool valid_date = (l_shipdate[i] >= 726350 && l_shipdate[i] <= 729313);
    bool valid_quantity = (l_quantity[i] < 70.0);
    bool valid_discount = (l_discount[i] >= 0.01 && l_discount[i] < 0.08);
    l_quantity[i] = (valid_date && valid_quantity && valid_discount) ? 1 : 0;
  }
}

void multiply_cpu(int n, double* l_quantity, double* l_extendedprice, double* l_discount)
{
  for (int i = 0; i < n; i++) {
    l_extendedprice[i] = (l_quantity[i]) ? l_extendedprice[i]*l_discount[i] : 0;
  }
}

template<class T>
void vec2ptr(std::vector<T> vec, T* ptr, int size)
{
  for (int i=0; i<size; i++) {
    ptr[i] = vec[i];
  }
}

template<typename T>
void bin2ptr(const char* filename, T **ptr)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // read the data
    char *temp = new char[fileSize];
    file.read(temp, fileSize);
    *ptr = reinterpret_cast<T*>(temp);
}

struct LineItem {
  double* l_quantity;
  double* l_extendedprice;
  double* l_discount;
  int* l_shipdate;
  int* size;
};

int dtoi(std::string str) {
  std::istringstream date(str);
  std::string time;
  int result = 0;
  int count = 0;
  while (getline(date, time, '-')) {
    int multiplier = 1;
    if (count==0) multiplier=365;
    if (count==1) multiplier=30;
    ++count;
    result += stoi(time)*multiplier;
  }
  return result;
}

void parse_lineitem(std::string path, LineItem& record, bool overwrite_file)
{ 
  bool file_exists = true;
  if (overwrite_file)
  {
    std::ofstream l_quantity ("cpp_tpch_q6/data/l_quantity.bin", std::ios::binary | std::ios::app);
    std::ofstream l_extendedprice ("cpp_tpch_q6/data/l_extendedprice.bin", std::ios::binary | std::ios::app);
    std::ofstream l_discount ("cpp_tpch_q6/data/l_discount.bin", std::ios::binary | std::ios::app);
    std::ofstream l_shipdate ("cpp_tpch_q6/data/l_shipdate.bin", std::ios::binary | std::ios::app);
    std::ofstream size ("cpp_tpch_q6/data/size.bin", std::ios::binary | std::ios::app);
    std::cout << "Parsing lineitem" << std::endl;

    std::fstream buffer(path);
    std::string line;
    int record_size = 0;
    while (getline(buffer, line)) {
      std::istringstream row(line);
      std::string field;
      int column;
      while (getline(row, field, DELIMITER)) {
        if (column==QUANTITY) {
          double q = std::stod(field);
          l_quantity.write(reinterpret_cast<char*>(&q), sizeof(double));
        } else if (column==EXTENDED_PRICE) {
          double q = std::stod(field);
          l_extendedprice.write(reinterpret_cast<char*>(&q), sizeof(double));
        } else if (column==DISCOUNT) {
          double q = std::stod(field);
          l_discount.write(reinterpret_cast<char*>(&q), sizeof(double));
        } else if (column==SHIPDATE) {
          int q = dtoi(field);
          l_shipdate.write(reinterpret_cast<char*>(&q), sizeof(int));
        }
        ++column;
        if (column==NUM_COLUMN) {
          column = 0;
          continue;
        }
      }
      ++record_size;
    }
    size.write(reinterpret_cast<char*>(&record_size), sizeof(int));
  }
  else
  {
    std::cout << "Table already parsed in binary, using that instead." << std::endl;
  }

  // Write binary to variables
  bin2ptr("cpp_tpch_q6/data/l_quantity.bin", &record.l_quantity);
  bin2ptr("cpp_tpch_q6/data/l_extendedprice.bin", &record.l_extendedprice);
  bin2ptr("cpp_tpch_q6/data/l_discount.bin", &record.l_discount);
  bin2ptr("cpp_tpch_q6/data/l_shipdate.bin", &record.l_shipdate);
  bin2ptr("cpp_tpch_q6/data/size.bin", &record.size);
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
  parse_lineitem(LINEITEM_PATH, lineitem, overwrite_file);

  std::cout << "Starting program" << std::endl;
  double* l_quantity;
  double* l_extendedprice;
  double* l_discount;
  int* l_shipdate;
  int N = *lineitem.size;

  std::cout << "Size: " << N << std::endl;
  std::cout << "l_quantity: " << lineitem.l_quantity[0] << std::endl;

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
  std::cout << "Amount: " << amount << std::endl;
  std::cout << "N: " << N << std::endl;

  // Free memory
  cudaFree(l_discount);
  cudaFree(l_extendedprice);
  cudaFree(l_shipdate);
  cudaFree(l_discount);
  
  return 0;
}
