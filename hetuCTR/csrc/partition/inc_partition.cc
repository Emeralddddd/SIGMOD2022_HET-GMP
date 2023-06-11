#include "inc_partition.h"

namespace hetuCTR {

std::unique_ptr<IncPartitionStruct> inc_partition(
  const py::array_t<int>& _input_data,
  const py::array_t<float>& _comm_mat,
  int n_part, int batch_size, float theta) {
  PYTHON_CHECK_ARRAY(_input_data);
  PYTHON_CHECK_ARRAY(_comm_mat);
  assert(_input_data.ndim() == 2);
  assert(_comm_mat.ndim() == 2);
  assert(_comm_mat.shape(0) == _comm_mat.shape(1));
  assert(_comm_mat.shape(0) == n_part);
  return  std::make_unique<IncPartitionStruct>(_input_data, _comm_mat, n_part, batch_size, theta);
}

} //namespace hetuCTR
