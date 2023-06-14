aux_source_directory(utils utils_src)
aux_source_directory(pybind pybind_src)
aux_source_directory(core core_src)

pybind11_add_module(hetuCTR ${pybind_src} ${utils_src} ${core_src})

target_include_directories(hetuCTR
    PRIVATE
    .
    ${cub_SOURCE_DIR}
    ${thrust_SOURCE_DIR}
    ${NCCL_INCLUDE_DIRS}
)

target_link_libraries(hetuCTR
    PRIVATE
    CUDA::cudart
    ${NCCL_LIBRARIES}
    ZMQ::lib
)

find_package(OpenMP REQUIRED)
set(Protobuf_INCLUDE_DIR "/usr/local/protobuf/include")
set(Protobuf_LIBRARY "/usr/local/protobuf/lib")
find_package(Protobuf REQUIRED)
# include_directories(${Protobuf_INCLUDE_DIRS})

# set(PROTO_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto)
set(PROTO_FILES proto/partial_result.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

aux_source_directory(partition partition_src)

pybind11_add_module(hetuCTR_partition ${partition_src} ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(hetuCTR_partition PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(hetuCTR_partition PRIVATE /usr/local/protobuf/lib/libprotobuf.so /usr/local/protobuf/lib/libprotobuf.a)
target_include_directories(hetuCTR_partition PRIVATE . partition)
target_include_directories(hetuCTR_partition PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(hetuCTR_partition PRIVATE ${Protobuf_INCLUDE_DIRS})
