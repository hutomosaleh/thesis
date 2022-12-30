#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <cuda_runtime_api.h>

#include "task.h"

class TaskManager
{
  public:
    void run(int type);
    void start_host_consumer();
    void start_device_consumer();
    void start_hybrid_consumer();
    void read_stats();

    TaskManager(std::deque<Task*> queue, int loops);

  private:
    bool _pop_task(Task** task);

    int _loops;
    int _queue_size;
    std::deque<Task*> _queue;
    cudaStream_t* _streams = nullptr;
    std::atomic<int> _current_index = {0};
    std::atomic<int> _hits = {0};
    std::atomic<int> _gpu_time = {0};
    std::atomic<int> _gpu_calls = {0};
    std::atomic<int> _cpu_time = {0};
    std::atomic<int> _cpu_calls = {0};
    std::atomic<double> _result = {0};
    std::vector<std::thread> _thread_pool;
};
