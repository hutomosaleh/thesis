#include "task_generator.hpp"

#include <deque>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>

#include "task.h"
#include "defs.hpp"
#include "data_types.hpp"
#include "parser.hpp"


TaskGenerator::TaskGenerator(int size) : _task_size(size) {}

void TaskGenerator::_add_to_queue(Task task)
{
  _queue.push_front(task);
}

std::deque<Task> TaskGenerator::generate(bool overwrite_file)
{
  Parser p;
  p.parse(LINEITEM_PATH, _lineitem, overwrite_file);

  // Divide lineitem into chunks of tasks
  int id = 0;
  std::cout << "Generating tasks" << std::endl;
  for(int i=0; i < _lineitem.size[0]; i+=_task_size)
  {
    Task task;
    for(int j=0; j< _task_size; j++)
    {
      TupleQ6 tuple {
        _lineitem.l_quantity[i+j],
        _lineitem.l_extendedprice[i+j],
        _lineitem.l_discount[i+j],
        _lineitem.l_shipdate[i+j]
      };
      task.add(tuple);
    }
    task.set_id(++id);
    _add_to_queue(task);
  }
  std::cout << "Tasks generated" << std::endl;
  return _queue;
}
