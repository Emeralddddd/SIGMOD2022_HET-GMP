#include "inc_partition.h"

namespace hetuCTR{
class MultiPartitionStruct{
public:
    MultiPartitionStruct(int n_part) : n_part_(n_part){}
    void AddNewData(const py::array_t<int>& _input_data,
        const py::array_t<float>& _comm_mat,
        int n_part, int batch_size, float theta);
    void RunPartition(int iter_nums);
    void RunPartition();
    void savePartialResult(std::string path);

private:
    std::unique_ptr<IncPartitionStruct> partitioner_;
    std::vector<int> embed_partiton_;
    int n_part_, max_embed_;
};

std::unique_ptr<MultiPartitionStruct> get_multi_partition(
  int n_part);
}
