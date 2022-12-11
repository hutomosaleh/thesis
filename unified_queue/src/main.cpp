#include <iostream>

#include "task_generator.h"
#include "data_types.hpp"

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
  
  TaskGenerator task;
  task.run(r, overwrite_file);
  return 0;
}
