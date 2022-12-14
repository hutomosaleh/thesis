#include "task_manager.hpp"

#include <chrono>
#include <deque>
#include <ios>
#include <mutex>
#include <thread>
#include <iostream>

#include "defs.hpp"
#include "task.h"

TaskManager::TaskManager(std::deque<Task> queue) : _queue(queue) {};

void TaskManager::read_stats()
{
  std::cout << "\n ==== Task Manager Stats ====" << std::endl;
  std::cout << "CPU Calls: " << _cpu_calls << std::endl;
  std::cout << "CPU Total Time: " << _cpu_time << "ms" << std::endl;
  std::cout << "GPU Calls: " << _gpu_calls << std::endl;
  std::cout << "GPU Total Time: " << _gpu_time << "ms" << std::endl;
  std::cout << "Total Time: " << _gpu_time + _cpu_time << "ms" << std::endl;
  std::cout << "Result: " << std::fixed << _result << std::endl;
  std::cout << "Hits: " << _hits << std::endl;
  std::cout << "Total tuples: " << TASK_SIZE*(_cpu_calls+_gpu_calls) << std::endl;

}

bool TaskManager::_pop_task(Task &task)
{
  bool success = false;
  _m.lock();
  if (!_queue.empty())
  {
    task = _queue.back();
    _queue.pop_back();
    success = true;
  }
  _m.unlock();
  return success;
}

void TaskManager::start_device_consumer()
{
  Task task;
  while (true)
  {
    if (!_pop_task(task))
    {
      std::cout << "Queue is empty, stopping device GPU thread" << std::endl;
      break;
    }
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
    if (!_pop_task(task))
    {
      std::cout << "Queue is empty, stopping device CPU thread" << std::endl;
      break;
    }
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
  std::vector<std::thread> threads;
  switch (type)
  {
    case CPU_TASK:
      threads.push_back(std::thread(&TaskManager::start_host_consumer, this));
      break;
    case GPU_TASK:
      threads.push_back(std::thread(&TaskManager::start_device_consumer, this));
      break;
    default:
      threads.push_back(std::thread(&TaskManager::start_host_consumer, this));
      threads.push_back(std::thread(&TaskManager::start_device_consumer, this));
  }

  // Join all threads
  for (auto &t : threads)
  {
    t.join();
  }
}
