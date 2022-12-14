#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>

#include "task.h"

class TaskManager
{
  public:
    void run();
    void start_host_consumer();
    void start_device_consumer();

    TaskManager(std::deque<Task>);

  private:
    bool _pop_task(Task&);

    std::mutex _m;
    std::deque<Task> _queue;
    std::atomic<int> _hits = {0};
    std::atomic<double> _result = {0};
    std::vector<std::thread> _thread_pool;
    bool _gpu_idle;
    bool _cpu_idle;

};
