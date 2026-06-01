set(CMAKE_POSITION_INDEPENDENT_CODE ON)
cmake_minimum_required(VERSION 3.18)  
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(TreeBuilder LANGUAGES CXX CUDA)  
add_subdirectory(/root/autodl-tmp/CTS/src/tools)
add_executable(TreeBuilder
    TreeBuilder.cc
)
target_link_libraries(TreeBuilder PUBLIC  gcts_tools)
target_include_directories(TreeBuilder PUBLIC 
    /root/autodl-tmp/CTS/src/include
    /root/autodl-tmp/CTS/src/include/databse
   )

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(gcts_tools PRIVATE -fconcepts)  # 确保开启 Concepts 支持
endif()

target_compile_options(TreeBuilder PRIVATE -O3 -march=native)


add_executable(Cluster
    Cluster.cc
)
target_link_libraries(Cluster PUBLIC  gcts_tools)
target_include_directories(Cluster PUBLIC 
    /root/autodl-tmp/CTS/src/include
    /root/autodl-tmp/CTS/src/include/databse
   )

target_compile_options(Cluster PRIVATE -O3 -march=native)

add_executable(flow
    flow.cc
)
target_link_libraries(flow PUBLIC  gcts_tools)
target_include_directories(flow PUBLIC 
    /root/autodl-tmp/CTS/src/include
    /root/autodl-tmp/CTS/src/include/databse
   )

target_compile_options(flow PRIVATE -O3 -march=x86-64)

