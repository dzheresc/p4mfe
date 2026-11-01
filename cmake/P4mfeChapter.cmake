# p4mfe_add_module(<name> [DEPS <targets>...])
#
# Declares one module of the companion (a chapter or `common`) laid out as:
#   <dir>/include/p4mfe/...   public headers
#   <dir>/src/*.cpp           optional sources (header-only if absent)
#   <dir>/tests/*.cpp         GoogleTest unit tests
#   <dir>/examples/*.cpp      optional small programs, one executable each
#   <dir>/bench/*.cpp         optional Google Benchmark programs
#
# Creates the library target p4mfe_<name> (alias p4mfe::<name>), the test
# executable p4mfe_<name>_tests, one executable per example, and, with
# P4MFE_BUILD_BENCHMARKS, the benchmark executable p4mfe_<name>_bench.
function(p4mfe_add_module name)
  cmake_parse_arguments(ARG "" "" "DEPS" ${ARGN})
  set(target p4mfe_${name})

  file(GLOB srcs CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  if(srcs)
    add_library(${target} ${srcs})
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${target} PUBLIC ${ARG_DEPS})
  else()
    add_library(${target} INTERFACE)
    target_include_directories(${target} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${target} INTERFACE ${ARG_DEPS})
  endif()
  add_library(p4mfe::${name} ALIAS ${target})

  file(GLOB examples CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp)
  foreach(ex ${examples})
    get_filename_component(ex_name ${ex} NAME_WE)
    add_executable(${name}_${ex_name} ${ex})
    target_link_libraries(${name}_${ex_name} PRIVATE ${target})
  endforeach()

  if(P4MFE_BUILD_TESTS)
    file(GLOB tests CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
    if(tests)
      add_executable(${target}_tests ${tests})
      target_link_libraries(${target}_tests PRIVATE ${target} GTest::gtest_main)
      gtest_discover_tests(${target}_tests)
    endif()
  endif()

  if(P4MFE_BUILD_BENCHMARKS)
    file(GLOB benches CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cpp)
    if(benches)
      add_executable(${target}_bench ${benches})
      target_link_libraries(${target}_bench PRIVATE ${target} p4mfe::bench_support)
      set_property(GLOBAL APPEND PROPERTY P4MFE_BENCH_TARGETS ${target}_bench)
    endif()
  endif()
endfunction()
