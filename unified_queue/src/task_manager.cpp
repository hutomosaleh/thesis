#include "task_manager.hpp"

#include <deque>
#include <ios>
#include <mutex>
#include <thread>
#include <iostream>

#include "defs.hpp"
#include "task.h"

TaskManager::TaskManager(std::deque<Task> queue) : _queue(queue) {};

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
    task.consume(GPU_TASK);
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
    task.consume(CPU_TASK);
    _hits += task.get_hits();
    for (double r = _result; !_result.compare_exchange_weak(r, r+task.get_result()););
  }
}

void TaskManager::run()
{
  std::thread threads[2];

  // Spawn CPU Thread
  std::cout << "Spawning cpu thread" << std::endl;
  threads[0] = std::thread(&TaskManager::start_host_consumer, this);

  // Spawn GPU Thread
  std::cout << "Spawning gpu thread" << std::endl;
  threads[1] = std::thread(&TaskManager::start_device_consumer, this);

  // Join all threads
  for (auto &t : threads)
  {
    t.join();
  }

  // Return results
  std::cout << "Tasks done, result: " << std::fixed << _result << std::endl;
  std::cout << "Tasks done, hits: " << _hits << std::endl;
}
