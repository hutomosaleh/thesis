#include <deque>
#include <iostream>

#include "defs.hpp"
#include "data_types.hpp"
#include "task_generator.hpp"
#include "task_manager.hpp"

int main(int argc, char** argv)
{
  std::cout << "\n ==== Running Unified Queue ====" << std::endl;
  int type = 2;
  bool overwrite_file = false;
  if (argc > 1) {
    type = atof(argv[1]);
    std::cout << "Type: " << type << std::endl;
    if (argc > 2)
    {
      std::string str(argv[2]);
      if (str == "overwrite") overwrite_file = true;
    }
  } else { std::cout << "Type set to default: " << type << std::endl; }
  std::cout << "0 : CPU | 1: GPU | Else: Hybrid" << std::endl;
  
  int task_size = TASK_SIZE;

  // Generate tasks
  TaskGenerator task(task_size);
  std::deque<Task> queue = task.generate(overwrite_file);

  // Start consuming tasks
  TaskManager manager(queue);
  manager.run(type);
  manager.read_stats();
  return 0;
}
