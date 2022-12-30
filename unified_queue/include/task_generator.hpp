#pragma once

#include <deque>

#include "data_types.hpp"
#include "task.h"

class TaskGenerator
{
  public:
    std::deque<Task*> generate(bool new_tbl);

    TaskGenerator(int size);
    ~TaskGenerator();
  
  private:
    void _add_to_queue(Task* task);

    LineItem _lineitem;
    std::deque<Task*> _queue;
    int _task_size;
};
