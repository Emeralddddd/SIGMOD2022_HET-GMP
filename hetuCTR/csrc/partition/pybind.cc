#include "pybind/pybind.h"
#include "partition.h"
#include "inc_partition.h"
#include "multi_partition.h"

namespace hetuCTR {

PYBIND11_MODULE(hetuCTR_partition, m) {
  m.doc() = "hetuCTR graph partition C++ implementation"; // module docstring
  py::class_<PartitionStruct>(m, "_PartitionStruct", py::module_local())
    .def("refine_data", &PartitionStruct::refineData)
    .def("refine_embed", &PartitionStruct::refineEmbed)
    .def("get_communication", &PartitionStruct::getCommunication)
    .def("get_priority", &PartitionStruct::getPriority)
    .def("get_result", [](PartitionStruct &s) {
      return py::make_tuple(bind::vec_nocp(s.res_data_), bind::vec_nocp(s.res_embed_));
    })
    .def("get_data_cnt", [](PartitionStruct &s) {
      return bind::vec_nocp(s.cnt_data_);
    })
    .def("get_embed_cnt", [](PartitionStruct &s) {
      return bind::vec_nocp(s.cnt_embed_);
    });
  m.def("partition", partition);

  py::class_<IncPartitionStruct>(m, "_IncPartitionStruct", py::module_local())
    .def("refine_data", &PartitionStruct::refineData)
    .def("refine_embed", &PartitionStruct::refineEmbed)
    .def("get_communication", &PartitionStruct::getCommunication)
    .def("get_priority", &IncPartitionStruct::getPriority)
    .def("get_result", &IncPartitionStruct::getResult)
    .def("get_data_cnt", [](IncPartitionStruct &s) {
      return bind::vec_nocp(s.cnt_data_);
    })
    .def("get_embed_cnt", [](IncPartitionStruct &s) {
      return bind::vec_nocp(s.cnt_embed_);
    }).def("save_partial_result", &IncPartitionStruct::savePartialResult);
  m.def("inc_partition", inc_partition);

  py::class_<MultiPartitionStruct>(m, "_MultiPartitionStruct", py::module_local())
    .def("add_new_data", &MultiPartitionStruct::AddNewData)
    .def("run_partitoin", &MultiPartitionStruct::RunPartition)
    .def("save_partial_result", &MultiPartitionStruct::savePartialResult);

  m.def("get_multi_partition", get_multi_partition);
} // PYBIND11_MODULE

} // namespace hetuCTR
