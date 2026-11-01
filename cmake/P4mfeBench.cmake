# p4mfe_add_bench_run_target()
#
# Adds the `bench` target, which runs every chapter's benchmark executable and
# writes one Google Benchmark JSON file per executable to <build>/bench/.
# The size tier comes from the P4MFE_BENCH_TIER environment variable
# (smoke | standard | large, default standard); any Google Benchmark flag can
# be set through its environment variable, e.g. BENCHMARK_MIN_TIME=0.01s.
function(p4mfe_add_bench_run_target)
  get_property(benches GLOBAL PROPERTY P4MFE_BENCH_TARGETS)
  set(out_dir ${CMAKE_BINARY_DIR}/bench)
  set(commands)
  foreach(b ${benches})
    list(APPEND commands
      COMMAND $<TARGET_FILE:${b}>
              --benchmark_out=${out_dir}/${b}.json
              --benchmark_out_format=json)
  endforeach()
  add_custom_target(bench
    COMMAND ${CMAKE_COMMAND} -E make_directory ${out_dir}
    ${commands}
    DEPENDS ${benches}
    USES_TERMINAL
    COMMENT "Running p4mfe benchmarks (JSON in ${out_dir})")
endfunction()
