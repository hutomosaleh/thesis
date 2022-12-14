#pragma once

#include <vector>
#include <string>

#include "data_types.hpp"

class Task
{
  public:
    void set_id(int);
    void add(TupleQ6);
    void consume(int);
    int get_hits();
    double get_result();

  private:
    int _id = 0;
    int _size = 0;
    int _hits = 0;
    double _result = 0;
    std::vector<TupleQ6> _data;
};
