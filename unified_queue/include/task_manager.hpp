#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>

#include "task.h"

class TaskManager
{
  public:
    void run(int type);
    void start_host_consumer();
    void start_device_consumer();
    void read_stats();

    TaskManager(std::deque<Task> queue);

  private:
    bool _pop_task(Task&);

    std::mutex _m;
    int _queue_size;
    std::deque<Task> _queue;
    std::atomic<int> _current_index = {0};
    std::atomic<int> _hits = {0};
    std::atomic<int> _gpu_time = {0};
    std::atomic<int> _gpu_calls = {0};
    std::atomic<int> _cpu_time = {0};
    std::atomic<int> _cpu_calls = {0};
    std::atomic<double> _result = {0};
    std::vector<std::thread> _thread_pool;
};
