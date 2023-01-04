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

TaskGenerator::~TaskGenerator()
{
  if (_lineitem.size != nullptr) _delete_lineitem();
  if (!_queue.empty()) for (auto task : _queue) delete task;
}

void TaskGenerator::_add_to_queue(Task* task)
{
  _queue.push_front(task);
}

std::deque<Task*> TaskGenerator::generate(bool new_tbl)
{
  Parser p;
  p.parse(LINEITEM_PATH, _lineitem, new_tbl);

  // Divide lineitem into chunks of tasks
  int id = 0;
  std::cout << "Generating tasks" << std::endl;
  std::cout << "Queue size: " << (int)_queue.size() << std::endl;
  for(int i=0; i < _lineitem.size[0]-_task_size; i+=_task_size)
  {
    Task* task = new Task(_task_size);
    for(int j=0; j< _task_size; j++)
    {
      TupleQ6 tuple {
        _lineitem.l_quantity[i+j],
        _lineitem.l_extendedprice[i+j],
        _lineitem.l_discount[i+j],
        _lineitem.l_shipdate[i+j]
      };
      task->add(tuple);
    }
    task->set_id(++id);
    _add_to_queue(task);
  }
  std::cout << "Tasks generated" << std::endl;
  std::cout << "Queue size: " << (int)_queue.size() << std::endl;
  return _queue;
}

void TaskGenerator::_delete_lineitem()
{
  delete[] _lineitem.l_discount;
  delete[] _lineitem.l_quantity;
  delete[] _lineitem.l_shipdate;
  delete[] _lineitem.l_extendedprice;
  delete[] _lineitem.size;
}
