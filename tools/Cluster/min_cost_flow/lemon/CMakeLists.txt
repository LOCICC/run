include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}  # 当前目录（min_cost_flow/）
    ${CMAKE_CURRENT_SOURCE_DIR}/lemon  # lemon头文件目录
    ${CMAKE_CURRENT_BINARY_DIR}  # 生成的config.h可能在这里
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/config.h)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/lemon.pc.in
               ${CMAKE_CURRENT_BINARY_DIR}/lemon.pc @ONLY)

set(LEMON_SOURCES
    arg_parser.cc
    base.cc
    color.cc
    lp_base.cc
    lp_skeleton.cc
    random.cc
    bits/windows.cc)

add_library(lemon ${LEMON_SOURCES})

target_link_libraries(lemon ${GLPK_LIBRARIES} ${COIN_LIBRARIES}
                      ${ILOG_LIBRARIES} ${SOPLEX_LIBRARIES})
target_include_directories(lemon PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lemon>
    $<INSTALL_INTERFACE:include>
)

if(UNIX)
  set_target_properties(
    lemon
    PROPERTIES OUTPUT_NAME emon
               VERSION ${LEMON_VERSION}
               SOVERSION ${LEMON_VERSION})
endif()
