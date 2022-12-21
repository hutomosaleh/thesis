#include <deque>
#include <iostream>

#include "defs.hpp"
#include "data_types.hpp"
#include "task_generator.hpp"
#include "task_manager.hpp"

int main(int argc, char** argv)
{
  std::cout << "\n ==== Running Unified Queue ====" << std::endl;
  int type = (argc > 1) ? atoi(argv[1]) : 2;
  bool overwrite_file = (argc > 2 && std::string(argv[2]) == "overwrite") ? true : false;
  int loop_count = (argc > 3 && atoi(argv[3]) != 0) ? atoi(argv[3]) : 1;
  std::cout << "Type: " << type << " | Loop count: " << loop_count << std::endl;
  std::cout << "0 : CPU | 1: GPU | Else: Hybrid" << std::endl;
  
  int task_size = TASK_SIZE;

  // Generate tasks
  TaskGenerator task(task_size);
  std::deque<Task> queue = task.generate(overwrite_file);

  // Start consuming tasks
  TaskManager manager(queue, loop_count);
  manager.run(type);
  manager.read_stats();
  return 0;
}
