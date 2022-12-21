#pragma once

#include <vector>
#include <string>
#include <cuda_runtime_api.h>

#include "data_types.hpp"

class Task
{
  public:
    void set_id(int id);
    void add(TupleQ6);
    void consume(int type, cudaStream_t stream=cudaStreamLegacy);
    int get_hits();
    double get_result();

  private:
    int _id = 0;
    int _size = 0;
    int _hits = 0;
    double _result = 0;
    std::vector<TupleQ6> _data;
};
