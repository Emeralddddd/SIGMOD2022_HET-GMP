#pragma once

#include "partition.h"
#include "partial_result.pb.h"

namespace hetuCTR {
class IncPartitionStruct :public PartitionStruct {
public:
    IncPartitionStruct(const py::array_t<int>& _input_data, const py::array_t<float>& _comm_mat, int _n_part, int _batch_size, float _theta) : \
        PartitionStruct(_input_data, _comm_mat, _n_part, _batch_size, _theta) {}
    void savePartialResult(std::string);
};

std::unique_ptr<IncPartitionStruct> inc_partition(
  const py::array_t<int>& _input_data,
  const py::array_t<float>& _comm_mat,
  int n_part, int batch_size, float theta
);

}// namespace hetuCTR