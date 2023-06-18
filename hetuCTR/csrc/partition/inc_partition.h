#pragma once

#include "partition.h"
#include "partial_result.pb.h"

namespace hetuCTR {
class IncPartitionStruct :public PartitionStruct {
public:
    IncPartitionStruct(const py::array_t<int>& _input_data, const py::array_t<float>& _comm_mat, int _n_part, int _batch_size, float _theta);
    void savePartialResult(std::string);
    std::vector<int> res_embed_remaped_;
    std::unordered_map<int,int> g2l_, l2g_;
    py::array_t<float> getPriority();
    py::tuple getResult();
    int max_embed_;
};

std::unique_ptr<IncPartitionStruct> inc_partition(
  const py::array_t<int>& _input_data,
  const py::array_t<float>& _comm_mat,
  int n_part, int batch_size, float theta
);
}// namespace hetuCTR