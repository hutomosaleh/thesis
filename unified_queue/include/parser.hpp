#pragma once

#include <string>

#include "data_types.hpp"

class Parser
{
  public:
    void parse(std::string path, LineItem& record, bool overwrite_file);

  private:
    int dtoi(std::string date);
};
