if(DEBUG_ICTS_MIN_COST_FLOW)
  message(STATUS "CTS: DEBUG_ICTS_MIN_COST_FLOW")
  set(CMAKE_BUILD_TYPE "Debug")
else()
  message(STATUS "CTS: RELEASE_ICTS_MIN_COST_FLOW")
  set(CMAKE_BUILD_TYPE "Release")
endif()
add_subdirectory(/root/autodl-tmp/CTS/src/tools/Cluster/min_cost_flow/lemon)
add_library(ccts_min_cost_flow INTERFACE)
if(NOT TARGET lemon)
    message(FATAL_ERROR "LEMON library target not found")
endif()
target_link_libraries(ccts_min_cost_flow INTERFACE lemon)

target_include_directories(
  ccts_min_cost_flow
  INTERFACE /root/autodl-tmp/CTS/src/tools/min_cost_flow
  /root/autodl-tmp/CTS/src/tools/min_cost_flow/lemon
  )
