#pragma once

#include <deque>

#include "data_types.hpp"
#include "task.h"

class TaskGenerator
{
  public:
    std::deque<Task> generate(bool);

    TaskGenerator(int);
  
  private:
    void _add_to_queue(Task);

    LineItem _lineitem;
    std::deque<Task> _queue;
    int _task_size;
};
