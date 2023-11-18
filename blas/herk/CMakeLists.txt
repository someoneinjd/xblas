cmake_minimum_required(VERSION 3.10)

project(herk)
set(KERNEL             "herk")
set(PRECISIONS         "cs" "zd")
set(MATMULS            "cccsmatmul" "zzzdmatmul")
set(MAX_TEST_ID        4)

get_filename_component(CURRENT_KERNEL_PATH        "${CMAKE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(RECONFIGURABLE_MATMUL_PATH "${CURRENT_KERNEL_PATH}/../reconfigurable_matmul"  ABSOLUTE)

# Create targets for generating OneAPI files, reporting of resource usage, and synthesizing bitstreams.
include(${RECONFIGURABLE_MATMUL_PATH}/CMakeLists.txt)
