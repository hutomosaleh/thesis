#include "task_manager.hpp"

#include <chrono>
#include <deque>
#include <ios>
#include <iostream>
#include <thread>
#include <omp.h>
#include <cuda_runtime_api.h>

#include "defs.hpp"
#include "task.h"

TaskManager::TaskManager(std::deque<Task*> queue, int loops, int task_size) : _loops(loops), _queue_size(queue.size()), _task_size(task_size), _queue(queue) {};

static double get_avg(int time, int rep)
{
  double time_ms = (double) time / 1e3;
  return time_ms / rep;
}

void TaskManager::read_stats()
{
  double task_size_mb = 32*_task_size/1e6;
  int total_calls = _cpu_calls+_gpu_calls;
  std::cout << "\n ==== Task Manager Stats ====" << std::endl;
  std::cout << "Tuple size: 32 Bytes" << std::endl;
  std::cout << "Task size: " << task_size_mb << " MB" << std::endl;
  std::cout << "CPU|GPU calls: " << _cpu_calls << "|" << _gpu_calls << std::endl;
  std::cout << "Total Time: " << total_calls / 1e3 << " ms" << std::endl;
  std::cout << "CPU Time Avg: " << get_avg(_cpu_time, _cpu_calls) << " ms / call" << std::endl;
  std::cout << "GPU Time Avg: " << get_avg(_gpu_time, _gpu_calls) << " ms / call" << std::endl;
  std::cout << "Total Time Avg: " << get_avg(_gpu_time+_cpu_time, _loops) << " ms / loop" << std::endl;
  std::cout << "Throughput Avg: " << task_size_mb * total_calls/_loops / get_avg(_gpu_time+_cpu_time, _loops)*1e3 << " MBps / loop" << std::endl;
  std::cout << "Result: " << std::fixed << _result << std::endl;
  std::cout << "Hits: " << _hits << std::endl;
  std::cout << "Total tuples: " << _task_size*total_calls/_loops << std::endl;
}

bool TaskManager::_pop_task(Task** task)
{
  bool success = false;
  if (_current_index < (int)_queue.size())
  {
    *task = _queue[_current_index++];
    success = true;
  }
  return success;
}

void TaskManager::start_hybrid_consumer()
{
  Task* task;
  Task* task_gpu;
  while (true)
  {
    if (!_pop_task(&task_gpu)) break;
    _gpu_calls++;
    auto start_gpu = std::chrono::steady_clock::now();
    task_gpu->consume(GPU_TASK);
    _gpu_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start_gpu).count();
    _hits += task_gpu->get_hits();
    for (double r = _result; !_result.compare_exchange_weak(r, r+task_gpu->get_result()););
    if (!_pop_task(&task)) break;
    _cpu_calls++;
    auto start = std::chrono::steady_clock::now(); 
    task->consume(CPU_TASK);
    _cpu_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    _hits += task->get_hits();
    for (double r = _result; !_result.compare_exchange_weak(r, r+task->get_result()););
  }
}

void TaskManager::start_device_consumer()
{
  Task* task;
#ifdef CUDASTREAM
  _streams = new cudaStream_t[STREAM_NUM];
  for (int i=0; i<STREAM_NUM; i++)
  {
    cudaStreamCreate(&_streams[i]);
  }
#endif
  while (true)
  {
    if (!_pop_task(&task)) break;
    _gpu_calls++;
    auto start = std::chrono::steady_clock::now();
    task->consume(GPU_TASK, _streams);
    _gpu_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    _hits += task->get_hits();
    for (double r = _result; !_result.compare_exchange_weak(r, r+task->get_result()););
  }
  if (_streams != nullptr) for (int i=0; i<STREAM_NUM; i++) cudaStreamDestroy(_streams[i]);
}

void TaskManager::start_host_consumer()
{
  Task* task;
  while (true)
  {
    if (!_pop_task(&task)) break;
    _cpu_calls++;
    auto start = std::chrono::steady_clock::now();
    task->consume(CPU_TASK);
    _cpu_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    _hits += task->get_hits();
    for (double r = _result; !_result.compare_exchange_weak(r, r+task->get_result()););
  }
}

void TaskManager::run(int type)
{
  // Trigger cuda context creation
  cudaFree(0);
  for (int i=0; i<_loops; i++)
  {
    _hits = 0;
    _result = 0.0f;
    _current_index = 0;
    switch (type)
    {
      case CPU_TASK:
      {
        start_host_consumer();
        break;
      }
      case GPU_TASK:
      {
        start_device_consumer();
        break;
      }
      default:
      {
        start_hybrid_consumer();

        //std::thread threads[2];
        //threads[0] = std::thread(&TaskManager::start_host_consumer, this);
        //threads[1] = std::thread(&TaskManager::start_device_consumer, this);
        //for (int i=0; i<2; i++) threads[i].join();

        //#pragma omp parallel
        //{
        //  #pragma omp task
        //  {
        //    start_host_consumer();
        //  }
        //  #pragma omp task
        //  {
        //    start_device_consumer();
        //  }
        //}

        break;

      }
    }
  }
}
