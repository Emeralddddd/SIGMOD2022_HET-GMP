#include "multi_partition.h"
#include <algorithm>

namespace hetuCTR{
    
void MultiPartitionStruct::AddNewData(const py::array_t<int>& _input_data,
    const py::array_t<float>& _comm_mat,
    int n_part, int batch_size, float theta){
    partitioner_ = inc_partition(_input_data, _comm_mat, n_part, batch_size, theta);
    partitioner_ -> initEmbedPartition(embed_partiton_);
}

void MultiPartitionStruct::RunPartition(){
    constexpr int iter_nums = 2;
    for(int i = 0; i < iter_nums; i++){
        partitioner_ -> refineData();
        partitioner_ -> refineEmbed();
    }
    
    max_embed_ = std::max(max_embed_,partitioner_ -> max_embed_);
    embed_partiton_.resize(max_embed_, -1);

    for(int i = 0; i < partitioner_-> n_embed_; i++){
        embed_partiton_[partitioner_->l2g_[i]] = partitioner_->res_embed_[i];
    }
}

void MultiPartitionStruct::savePartialResult(std::string path){
    partitioner_ -> savePartialResult(path);
}

std::unique_ptr<MultiPartitionStruct> get_multi_partition(int n_part)
{
    return std::make_unique<MultiPartitionStruct>(n_part);
}
}

