cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES "75;80;86")

add_library(gcts_tools SHARED /root/autodl-tmp/CTS/src/tools/flow.cc )
add_subdirectory(/root/autodl-tmp/CTS/src/tools/model)
add_subdirectory(/root/autodl-tmp/CTS/src/tools/Timing)
add_subdirectory(/root/autodl-tmp/CTS/src/tools/BST)
add_subdirectory(/root/autodl-tmp/CTS/src/tools/CBS)
add_subdirectory(/root/autodl-tmp/CTS/src/tools/Cluster)
target_link_libraries(gcts_tools PUBLIC gcts_BST gcts_CBS gcts_cluster gcts_model)


