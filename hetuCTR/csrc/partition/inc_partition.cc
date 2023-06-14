#include "inc_partition.h"
#include "partial_result.pb.h"
#include <fstream>

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

void IncPartitionStruct::savePartialResult(std::string path){
  std::cout << path << std::endl;
  PartialResult pr;
  for(int i = 0; i < n_embed_; i++){
    int dist_i = l2g_[i];
    (*pr.mutable_partition_map())[dist_i] = res_embed_[i];
  }
  for (int i = 0; i < n_part_; i++) {
    PriorityMap* pm = pr.add_priority_maps();
    for (int j = 0 ; j < n_embed_; j++) {
      if (cnt_part_embed_[i][j] == 0) continue;
      int dist_j = l2g_[j];
      (*pm->mutable_map_field())[dist_j] = comm_mat_[i][res_embed_[j]] * std::pow(soft_cnt_[cnt_part_embed_[i][j]], 2l) *
        ((1.0 / (embed_indptr_[j + 1] - embed_indptr_[j])) + (1.0 / cnt_part_embed_[i][j]));
    }
  }
  std::string output;
  if (!pr.SerializeToString(&output)) {
    std::cout << "Failed to write partial result." << std::endl;
    return;
  }

  std::ofstream out_file(path, std::ios::binary);
  if (!out_file) {
    std::cout << "Failed to open output file." << std::endl;
    return;
  }
  if (!out_file.write(output.data(), output.size())) {
    std::cout << "Failed to write output file." << std::endl;
    return;
  }
  return;
}

} //namespace hetuCTR
