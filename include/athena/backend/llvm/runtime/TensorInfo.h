#pragma once

#include <cstdint>

struct TensorInfo {
  uint64_t virtAddr;
  int dataType;
  uint64_t dims;
  uint64_t* shape;
};
