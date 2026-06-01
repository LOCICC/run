cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 
add_library(gcts_model /root/autodl-tmp/CTS/src/tools/model/ModelFactory.cc)

find_package(Eigen3 QUIET REQUIRED)
message(STATUS "CTS: Eigen3 ${EIGEN3_INCLUDE_DIR}")
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-std=c++20>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>"
)
target_link_libraries(gcts_model PUBLIC Eigen3::Eigen)
target_include_directories(gcts_model PUBLIC /root/autodl-tmp/CTS/src/tools/model)

