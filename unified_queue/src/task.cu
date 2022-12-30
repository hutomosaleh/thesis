#include "task.h"

#include <atomic>
#include <cuda_runtime_api.h>
#include <driver_types.h>
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

void Task::consume(int type, cudaStream_t* streams)
{
  double* q;
  double* e;
  double* d;
  int* s;

  switch (type)
  {
#ifdef MALLOCHOST
    case CPU_TASK:

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
      cudaFreeHost(q);
      cudaFreeHost(e);
      cudaFreeHost(s);
      cudaFreeHost(d);

      break;
#else
    case CPU_TASK:

      q = (double*) malloc(_size*sizeof(double));
      e = (double*) malloc(_size*sizeof(double));
      d = (double*) malloc(_size*sizeof(double));
      s = (int*) malloc(_size*sizeof(int));
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
      free(q);
      free(e);
      free(s);
      free(d);

      break;
#endif
#ifdef MALLOCMANAGED
    case GPU_TASK:

      // Allocate device memory
      cudaMallocManaged(&q, _size*sizeof(double));
      cudaMallocManaged(&e, _size*sizeof(double));
      cudaMallocManaged(&d, _size*sizeof(double));
      cudaMallocManaged(&s, _size*sizeof(int));
      for(int i=0; i<_size; ++i)
      {
        q[i] = _data[i].quantity;
        e[i] = _data[i].extendedprice;
        d[i] = _data[i].discount;
        s[i] = _data[i].shipdate;
      }

      int block_number = (_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
      check<<<block_number, BLOCK_SIZE>>>(_size, q, s, d);
      multiply<<<block_number, BLOCK_SIZE>>>(_size, q, e, d);

      // Compare for query hits and update results
      for (int i = 0; i < _size; i++)
      {
        if (e[i])
        {
          _hits++;
          _result += e[i];
        }
      }

      // Release memory
      cudaFree(q);
      cudaFree(e);
      cudaFree(s);
      cudaFree(d);
      break;
#else
    case GPU_TASK:

      // Allocate host memory
      double* q_h = (double*)malloc(_size*sizeof(double));
      double* e_h = (double*)malloc(_size*sizeof(double));
      double* d_h = (double*)malloc(_size*sizeof(double));
      int* s_h = (int*)malloc(_size*sizeof(int));
      
      // Allocate device memory
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

      if (streams != nullptr)
      {
        const int stream_size = _size / STREAM_NUM;
        for (int i=0; i<STREAM_NUM; i++)
        {
          int offset = i * stream_size;
          // Copy host to device
          cudaMemcpyAsync(&q[offset], &q_h[offset], stream_size*sizeof(double), cudaMemcpyHostToDevice, streams[i]);
          cudaMemcpyAsync(&e[offset], &e_h[offset], stream_size*sizeof(double), cudaMemcpyHostToDevice, streams[i]);
          cudaMemcpyAsync(&d[offset], &d_h[offset], stream_size*sizeof(double), cudaMemcpyHostToDevice, streams[i]);
          cudaMemcpyAsync(&s[offset], &s_h[offset], stream_size*sizeof(int), cudaMemcpyHostToDevice, streams[i]);
          check<<<stream_size/BLOCK_SIZE, BLOCK_SIZE, 0, streams[i]>>>(_size, q, s, d);
          multiply<<<stream_size/BLOCK_SIZE, BLOCK_SIZE, 0, streams[i]>>>(_size, q, e, d);
        }
        for (int i = 0; i < STREAM_NUM; ++i)
        {
          int offset = i * stream_size;
          cudaMemcpyAsync(&e_h[offset], &e[offset], stream_size*sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
        }
      }
      else {
        int block_number = (_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // Copy host to device
        cudaMemcpy(q, q_h, _size*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(e, e_h, _size*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d, d_h, _size*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(s, s_h, _size*sizeof(int), cudaMemcpyHostToDevice);

        // Run kernels
        check<<<block_number, BLOCK_SIZE>>>(_size, q, s, d);
        multiply<<<block_number, BLOCK_SIZE>>>(_size, q, e, d);

        // Copy device result to host
        cudaMemcpy(e_h, e, _size*sizeof(double), cudaMemcpyDeviceToHost);
      }

      // Compare for query hits and update results
      for (int i = 0; i < _size; i++)
      {
        if (e_h[i])
        {
          _hits++;
          _result += e_h[i];
        }
      }

      // Release memory
      cudaFree(q);
      cudaFree(e);
      cudaFree(s);
      cudaFree(d);
      free(q_h);
      free(e_h);
      free(s_h);
      free(d_h);

      break;
#endif
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
