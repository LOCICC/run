cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
# 设置 CUDA 标准（使用 nvcc 支持的最高标准）
set(CMAKE_CUDA_STANDARD 17)  # CUDA 11.0+ 支持 C++17
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
add_library(gcts_aution
/root/autodl-tmp/CTS_3D/src/tools/Cluster/auction/auction6.cu
                    )
              
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-std=c++20>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>"
)
   
target_include_directories(gcts_aution PUBLIC 
     /root/autodl-tmp/CTS_3D/src/tools/Cluster/auction
    ${CUDA_INCLUDE_DIRS})
target_link_libraries(gcts_aution PUBLIC ${CUDA_LIBRARIES})


message(STATUS "Found CUDA: ${CUDA_VERSION}")
message(STATUS "CUDA Toolkit path: ${CUDA_TOOLKIT_ROOT_DIR}")