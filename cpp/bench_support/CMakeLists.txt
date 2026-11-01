# Shared benchmark support: the benchmark main (with heap allocation counting
# reported through benchmark::MemoryManager), size tiers, and market data
# generators. Every chapter's bench executable links p4mfe::bench_support.
#
# main.cpp is an OBJECT library so the replacement operator new/delete is
# always linked into each benchmark executable.
add_library(p4mfe_bench_support OBJECT src/main.cpp)
target_include_directories(p4mfe_bench_support PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(p4mfe_bench_support PUBLIC
  benchmark::benchmark
  p4mfe_common
  p4mfe_ch01_calculus_options
  p4mfe_ch02_integration_bonds
  p4mfe_ch03_probability_black_scholes
  p4mfe_ch05_newton_bootstrapping)
target_compile_definitions(p4mfe_bench_support PRIVATE P4MFE_BUILD_TYPE="$<CONFIG>")
add_library(p4mfe::bench_support ALIAS p4mfe_bench_support)
