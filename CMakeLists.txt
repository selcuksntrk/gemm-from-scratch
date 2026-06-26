cmake_minimum_required(VERSION 4.2)
project(GEMM)

set(CMAKE_CXX_STANDARD 20)

add_executable(GEMM
        include/gemm.hpp
        src/gemm_naive.cpp
        src/main.cpp)

target_include_directories(GEMM PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
