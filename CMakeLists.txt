cmake_minimum_required(VERSION 3.20)
project(p4mfe VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(P4MFE_BUILD_TESTS "Build unit tests" ON)
option(P4MFE_BUILD_BENCHMARKS "Build performance benchmarks (Google Benchmark)" OFF)

if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Wconversion)
endif()

if(P4MFE_BUILD_TESTS)
  enable_testing()
  include(cmake/GoogleTest.cmake)
endif()

if(P4MFE_BUILD_BENCHMARKS)
  include(cmake/GoogleBenchmark.cmake)
endif()

include(cmake/P4mfeChapter.cmake)

add_subdirectory(cpp)

if(P4MFE_BUILD_BENCHMARKS)
  include(cmake/P4mfeBench.cmake)
  p4mfe_add_bench_run_target()
endif()
