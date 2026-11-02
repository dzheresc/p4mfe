#!/usr/bin/env bash
# Checks that the benchmarks exercise every public function of the chapter
# headers (docs/performance-testing-plan.md, "Covering all code paths").
#
#   tools/bench/coverage.sh            # needs gcovr (pip install gcovr)
#
# Builds the bench-coverage preset (gcc --coverage), runs the smoke tier, and
# lists functions in cpp/*/include (exercises excluded) that no benchmark
# called. Exit status 1 if any is missing from bench/coverage-waivers.txt.
# The function list comes from the headers, not from gcov, so templates and
# inline functions that no benchmark instantiates are reported too.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
build=$root/build/bench-coverage
cd "$root"

cmake --preset bench-coverage >/dev/null
cmake --build --preset bench-coverage -j >/dev/null
find "$build" -name '*.gcda' -delete
for b in $(find "$build/cpp" -type f -name 'p4mfe_*_bench' -perm -u+x | sort); do
  P4MFE_BENCH_TIER=smoke BENCHMARK_MIN_TIME=1x "$b" >/dev/null 2>&1
done

gcovr -r "$root" $(find "$build/cpp" -type d -name 'p4mfe_*_bench.dir') \
  --filter 'cpp/[^/]+/include/' --exclude '.*/exercises/.*' \
  --json "$build/coverage.json" --txt "$build/coverage.txt" --txt-summary >/dev/null
tail -n 4 "$build/coverage.txt"
python3 "$root/tools/bench/uncovered.py" "$build/coverage.json" "$root/bench/coverage-waivers.txt" "$root"
