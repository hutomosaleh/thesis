#include "task_manager.hpp"

#include <chrono>
#include <deque>
#include <ios>
#include <iostream>
#include <thread>
#include <cuda_runtime_api.h>

#include "defs.hpp"
#include "task.h"

TaskManager::TaskManager(std::deque<Task> queue, int loops) : _loops(loops), _queue_size(queue.size()), _queue(queue) {};

void TaskManager::read_stats()
{
  std::cout << "\n ==== Task Manager Stats ====" << std::endl;
  std::cout << "CPU Calls: " << _cpu_calls << std::endl;
  std::cout << "CPU Total Time: " << _cpu_time << " ms" << std::endl;
  std::cout << "GPU Calls: " << _gpu_calls << std::endl;
  std::cout << "GPU Total Time: " << _gpu_time << " ms" << std::endl;
  std::cout << "Total Time: " << _gpu_time + _cpu_time << " ms" << std::endl;
  std::cout << "Total Time per loop avg: " << (double)(_gpu_time + _cpu_time) / _loops << " ms" << std::endl;
  std::cout << "Result: " << std::fixed << _result << std::endl;
  std::cout << "Hits: " << _hits << std::endl;
  std::cout << "Total tuples: " << TASK_SIZE*(_cpu_calls+_gpu_calls) << std::endl;

}

bool TaskManager::_pop_task(Task &task)
{
  bool success = false;
  if (_current_index < (int)_queue.size())
  {
    task = _queue[_current_index++];
    success = true;
  }
  return success;
}

void TaskManager::start_hybrid_consumer()
{
  Task task;
  Task task_gpu;
  while (true)
  {
    if (!_pop_task(task_gpu)) break;
    _gpu_calls++;
    auto start_gpu = std::chrono::steady_clock::now();
    task_gpu.consume(GPU_TASK);
    _gpu_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_gpu).count();
    _hits += task_gpu.get_hits();
    for (double r = _result; !_result.compare_exchange_weak(r, r+task_gpu.get_result()););
    if (!_pop_task(task)) break;
    _cpu_calls++;
    auto start = std::chrono::steady_clock::now(); 
    task.consume(CPU_TASK);
    _cpu_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    _hits += task.get_hits();
    for (double r = _result; !_result.compare_exchange_weak(r, r+task.get_result()););
  }
}

void TaskManager::start_device_consumer()
{
  Task task;
  while (true)
  {
    if (!_pop_task(task)) break;
    _gpu_calls++;
    auto start = std::chrono::steady_clock::now();
    task.consume(GPU_TASK);
    _gpu_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    _hits += task.get_hits();
    for (double r = _result; !_result.compare_exchange_weak(r, r+task.get_result()););
  }
}

void TaskManager::start_host_consumer()
{
  Task task;
  while (true)
  {
    if (!_pop_task(task)) break;
    _cpu_calls++;
    auto start = std::chrono::steady_clock::now();
    task.consume(CPU_TASK);
    _cpu_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    _hits += task.get_hits();
    for (double r = _result; !_result.compare_exchange_weak(r, r+task.get_result()););
  }
}

void TaskManager::run(int type)
{
  // Trigger cuda context creation
  cudaFree(0);
  for (int i=0; i<_loops; i++)
  {
    _current_index = 0;
    switch (type)
    {
      case CPU_TASK:
        start_host_consumer();
        break;
      case GPU_TASK:
        start_device_consumer();
        break;
      default:
        start_hybrid_consumer();
        break;
    }
  }
}
