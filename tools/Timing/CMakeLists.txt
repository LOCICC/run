cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
add_library(gcts_Timing
/root/autodl-tmp/CTS/src/tools/Timing/TimingUpdate.cc
                       )
              
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-std=c++20>"
)
                
target_include_directories(gcts_Timing PUBLIC 
    /root/autodl-tmp/CTS/src/include
    /root/autodl-tmp/CTS/src/include/databse
    /root/autodl-tmp/CTS/src/tools/model
    ${CUDA_INCLUDE_DIRS})
target_link_libraries(gcts_Timing PUBLIC ${CUDA_LIBRARIES})

target_compile_options(gcts_Timing PRIVATE -O3 -march=x86-64)
