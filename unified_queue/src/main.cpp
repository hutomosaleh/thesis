#include <deque>
#include <iostream>

#include "defs.hpp"
#include "data_types.hpp"
#include "task_generator.hpp"
#include "task_manager.hpp"

int main(int argc, char** argv)
{
  float r = 1.0;
  bool overwrite_file = false;
  if (argc > 1) {
    r = atof(argv[1]);
    std::cout << "Ratio: " << r << std::endl;
    if (argc > 2)
    {
      std::string str(argv[2]);
      if (str == "overwrite") overwrite_file = true;
    }
  } else { std::cout << "Ratio set to default: " << r << std::endl; }
  
  int task_size = TASK_SIZE;

  // Generate tasks
  TaskGenerator task(task_size);
  std::deque<Task> queue = task.generate(overwrite_file);

  // Start consuming tasks
  TaskManager manager(queue);
  manager.run();
  manager.read_stats();
  return 0;
}
