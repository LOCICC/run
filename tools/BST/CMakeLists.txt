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
add_library(gcts_BST 
/root/autodl-tmp/CTS/src/tools/BST/BoundSkewTree.cc
/root/autodl-tmp/CTS/src/tools/BST/BoundSkewTree.cu
/root/autodl-tmp/CTS/src/tools/BST/BoundSkewTreeR.cu
/root/autodl-tmp/CTS/src/tools/BST/BoundSkewTreeM.cu
/root/autodl-tmp/CTS/src/tools/BST/Timing.cu
/root/autodl-tmp/CTS/src/tools/BST/Instgpu.cc
                       )
              
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-std=c++20>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>"
)

# 对于需要 Concepts 的代码，添加条件编译
target_compile_definitions(gcts_tools PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:USE_CPP20_CONCEPTS=1>
)
                
target_include_directories(gcts_BST PUBLIC 
    /root/autodl-tmp/CTS/src/include
    /root/autodl-tmp/CTS/src/include/databse
    ${CUDA_INCLUDE_DIRS})
target_link_libraries(gcts_BST PUBLIC gcts_Timing ${CUDA_LIBRARIES})

target_compile_options(gcts_BST PRIVATE -O3 -march=x86-64)
message(STATUS "Found CUDA: ${CUDA_VERSION}")
message(STATUS "CUDA Toolkit path: ${CUDA_TOOLKIT_ROOT_DIR}")