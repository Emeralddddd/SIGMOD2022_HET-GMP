#include "inc_partition.h"
#include "partial_result.pb.h"
#include <fstream>

namespace hetuCTR {

static float quickpow(float a,int n) {
  float ans = 1, temp = a;
  while(n) {
    if(n&1) ans *= temp;
    n >>= 1;
    temp *= temp;
  }
  return ans;
}

IncPartitionStruct::IncPartitionStruct(const py::array_t<int>& _input_data, const py::array_t<float>& _comm_mat, int _n_part, int _batch_size, float _theta)
: PartitionStruct(_n_part,_batch_size,_theta) {
  n_data_ = _input_data.shape(0);
  n_slot_ = _input_data.shape(1);
  n_edge_ = n_data_ * n_slot_;
  const int *old_data = _input_data.data();
  std::vector<int> data(n_edge_);
  n_embed_ = -1;
  max_embed_ = 0;
  for (int i = 0 ; i < n_edge_; i++) {
    max_embed_ = std::max(max_embed_, old_data[i]);
    if(!g2l_.count(old_data[i])) {
      ++n_embed_;
      g2l_[old_data[i]] = n_embed_;
      l2g_[n_embed_] = old_data[i];
      data[i] = n_embed_;
    }else{
      data[i] = g2l_[old_data[i]];
    }
  }
  n_embed_++;
  max_embed_++;
  std::vector<int> count(n_embed_);
  data_indptr_.resize(n_data_ + 1);
  embed_indptr_.resize(n_embed_ + 1);
  data_indices_.resize(n_edge_);
  embed_indices_.resize(n_edge_);

  for (int i = 0; i <= n_data_; i++) data_indptr_[i] = i * n_slot_;
  for (int i = 0 ; i < n_edge_; i++) {
    count[data[i]]++;
    data_indices_[i] = data[i];
  }
  for (int i = 1;i <= n_embed_; i++) {
    embed_indptr_[i] = embed_indptr_[i-1] + count[i - 1];
    count[i - 1] = 0;
  }
  assert(embed_indptr_[n_embed_] == n_edge_);
  for (int i = 0 ; i < n_edge_; i++) {
    int data_id = i / n_slot_;
    int embed_id = data[i];
    embed_indices_[embed_indptr_[embed_id] + count[embed_id]] = data_id;
    count[embed_id]++;
  }
  //  initSoftLabel
  int max = n_data_ / n_part_;
  soft_cnt_.resize(n_data_, 1);
  for (int i = 0; i < max; i++) {
    soft_cnt_[i] = 1 - quickpow(1.0 - (float)i / (float)max, batch_size_);
    soft_cnt_[i] *= (float)max / (float)batch_size_;
  }
  // initResult
  res_data_.resize(n_data_);
  res_embed_.resize(n_embed_);
  cnt_data_.resize(n_part_, 0);
  cnt_embed_.resize(n_part_, 0);

  cnt_part_embed_.resize(n_part_, std::vector<int>(n_embed_, 0));

  for (int i = 0; i < n_data_; i++) {
    res_data_[i] = rand() % n_part_;
    cnt_data_[res_data_[i]]++;
    for (int j = data_indptr_[i]; j < data_indptr_[i+1]; j++) {
      cnt_part_embed_[res_data_[i]][data_indices_[j]]++;
    }
  }
  for (int i = 0; i < n_embed_; i++) {
    res_embed_[i] = rand() % n_part_;
    cnt_embed_[res_embed_[i]]++;
  }
  // init param
  alpha_ = -100.0 / (n_data_ / n_part_);
  beta_ = -100.0 / (n_embed_ / n_part_);
  // init communication matrix
  comm_mat_.resize(n_part_, std::vector<float>(n_part_));
  for (int i = 0; i < n_part_; i++)
    for (int j = 0; j < n_part_; j++) comm_mat_[i][j] = _comm_mat.at(i, j);
}

py::array_t<float> IncPartitionStruct::getPriority() {
  py::array_t<float> priority({n_part_, max_embed_});
  for (int i = 0; i < n_part_; i++) {
    for (int j = 0 ; j < max_embed_; j++) {
      priority.mutable_at(i, j) = 0;
    }
  }
  for (int i = 0; i < n_part_; i++) {
    for (int j = 0 ; j < n_embed_; j++) {
      int dist_j = l2g_[j];
      if (cnt_part_embed_[i][j] != 0)
        priority.mutable_at(i, dist_j) = comm_mat_[i][res_embed_[j]] * std::pow(soft_cnt_[cnt_part_embed_[i][j]], 2l) *
          ((1.0 / (embed_indptr_[j + 1] - embed_indptr_[j])) + (1.0 / cnt_part_embed_[i][j]));
    }
  }
  return priority;
}

py::tuple IncPartitionStruct::getResult(){
  res_embed_remaped_.resize(max_embed_,-1);
  for(int i = 0; i < n_embed_; i++){
    int dist_i = l2g_[i];
    res_embed_remaped_[dist_i] = res_embed_[i];
  }
  return py::make_tuple(bind::vec_nocp(res_data_), bind::vec_nocp(res_embed_remaped_));
}

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
