#include "pybind/pybind.h"
#include "partition.h"
#include "inc_partition.h"

namespace hetuCTR {

PYBIND11_MODULE(hetuCTR_partition, m) {
  m.doc() = "hetuCTR graph partition C++ implementation"; // module docstring
  py::class_<PartitionStruct>(m, "_PartitionStruct", py::module_local())
    .def("refine_data", &PartitionStruct::refineData)
    .def("refine_embed", &PartitionStruct::refineEmbed)
    .def("get_communication", &PartitionStruct::getCommunication)
    .def("get_priority", &PartitionStruct::getPriority)
    .def("get_result", &PartitionStruct::getResult)
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
    .def("get_priority", &PartitionStruct::getPriority)
    .def("get_result", &PartitionStruct::getResult)
    .def("get_data_cnt", [](IncPartitionStruct &s) {
      return bind::vec_nocp(s.cnt_data_);
    })
    .def("get_embed_cnt", [](IncPartitionStruct &s) {
      return bind::vec_nocp(s.cnt_embed_);
    });
  m.def("inc_partition", inc_partition);

} // PYBIND11_MODULE

} // namespace hetuCTR
