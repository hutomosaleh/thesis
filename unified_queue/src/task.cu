#include "task.h"

#include <atomic>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <vector>

#include "defs.hpp"
#include "data_types.hpp"
#include "cpu_kernels.hpp"
#include "gpu_kernels.h"

void Task::set_id(int id)
{
  _id = id;
}

void Task::add(TupleQ6 tuple)
{
  _data.push_back(tuple);
  ++_size;
}

void Task::consume(int type)
{
  double* q;
  double* e;
  double* d;
  int* s;

  switch (type)
  {
    case CPU_TASK:

      //std::cout << "cpu_consume start " << _id << std::endl;
      cudaMallocHost(&q, _size*sizeof(double));
      cudaMallocHost(&e, _size*sizeof(double));
      cudaMallocHost(&d, _size*sizeof(double));
      cudaMallocHost(&s, _size*sizeof(int));
      for(int i=0; i<_size; ++i)
      {
        q[i] = _data[i].quantity;
        e[i] = _data[i].extendedprice;
        d[i] = _data[i].discount;
        s[i] = _data[i].shipdate;
      }
      check_cpu(_size, q, s, d);
      multiply_cpu(_size, q, e, d);
      for (int i = 0; i < _size; i++)
      {
        if (e[i])
        {
          _hits++;
          _result += e[i];
        }
      }
      cudaFree(q);
      cudaFree(e);
      cudaFree(s);
      cudaFree(d);
      //std::cout << "cpu_consume end " << _id << std::endl;
    case GPU_TASK:

      std::cout << "gpu_consume start " << _id << std::endl;
      double* q_h = (double*)malloc(_size*sizeof(double));
      double* e_h = (double*)malloc(_size*sizeof(double));
      double* d_h = (double*)malloc(_size*sizeof(double));
      int* s_h = (int*)malloc(_size*sizeof(int));
      
      cudaMalloc(&q, _size*sizeof(double));
      cudaMalloc(&e, _size*sizeof(double));
      cudaMalloc(&d, _size*sizeof(double));
      cudaMalloc(&s, _size*sizeof(int));
      for(int i=0; i<_size; ++i)
      {
        q_h[i] = _data[i].quantity;
        e_h[i] = _data[i].extendedprice;
        d_h[i] = _data[i].discount;
        s_h[i] = _data[i].shipdate;
      }
      cudaMemcpy(q, q_h, _size*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(e, e_h, _size*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d, d_h, _size*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(s, s_h, _size*sizeof(int), cudaMemcpyHostToDevice);

      int blockSize = 128;
      int numBlocks = (_size + blockSize - 1) / blockSize;
      check<<<numBlocks, blockSize>>>(_size, q, s, d);
      multiply<<<numBlocks, blockSize>>>(_size, q, e, d);
      cudaDeviceSynchronize();
      cudaMemcpy(e, e_h, _size*sizeof(double), cudaMemcpyDeviceToHost);
      for (int i = 0; i < _size; i++)
      {
        if (e_h[i])
        {
          _hits++;
          _result += e_h[i];
        }
      }
      cudaFree(q);
      cudaFree(e);
      cudaFree(s);
      cudaFree(d);
      std::cout << "gpu_consume end " << _id << std::endl;
  }
}

int Task::get_hits()
{
  return _hits;
}

double Task::get_result()
{
  return _result;
}
