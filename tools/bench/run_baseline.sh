#!/usr/bin/env bash
# Records a performance baseline (docs/performance-testing-plan.md, "Baseline").
#
#   tools/bench/run_baseline.sh <out-dir> [repetitions]
#
# Builds the `bench` preset (Release), then
#   1. timing: every chapter benchmark at P4MFE_BENCH_TIER (default standard)
#      with <repetitions> repetitions (default 10), aggregates only
#      -> <out-dir>/cpp/*.json
#   2. heap: one run with exactly 4 iterations per benchmark, so the counting
#      allocator's numbers are deterministic -> <out-dir>/mem/*.json
#      (separate because Google Benchmark 1.9.1 keeps dangling pointers to the
#      memory results of earlier repetitions)
#   3. peak RSS: each benchmark once in its own process -> <out-dir>/rss.json
#      (includes the executable's market data, generated at startup)
#   4. writes <out-dir>/ENVIRONMENT.md (machine, OS, compiler, flags, commit)
#      and <out-dir>/SUMMARY.md (one table per executable).
#
# For a quiet local baseline, set the CPU governor to performance and pin the
# run, e.g.  taskset -c 2 tools/bench/run_baseline.sh bench/baselines/<date>-<machine>
set -euo pipefail

out=${1:?usage: run_baseline.sh <out-dir> [repetitions]}
reps=${2:-10}
tier=${P4MFE_BENCH_TIER:-standard}
root=$(cd "$(dirname "$0")/../.." && pwd)
build=$root/build/bench-release
here=$root/tools/bench

mkdir -p "$out/cpp" "$out/mem"
out=$(cd "$out" && pwd)

cmake --preset bench -S "$root" >/dev/null
cmake --build --preset bench -j >/dev/null

mapfile -t benches < <(find "$build/cpp" -type f -name 'p4mfe_*_bench' -perm -u+x | sort)
echo "Found ${#benches[@]} benchmark executables; tier=$tier, repetitions=$reps"

for b in "${benches[@]}"; do
  name=$(basename "$b")
  echo "== $name"
  P4MFE_BENCH_TIER=$tier "$b" \
    --benchmark_repetitions="$reps" \
    --benchmark_report_aggregates_only=true \
    --benchmark_out="$out/cpp/$name.json" \
    --benchmark_out_format=json >/dev/null 2>"$out/cpp/$name.stderr"
  P4MFE_BENCH_TIER=$tier BENCHMARK_MIN_TIME=4x "$b" \
    --benchmark_out="$out/mem/$name.json" \
    --benchmark_out_format=json >/dev/null 2>>"$out/cpp/$name.stderr"
done

echo "== peak RSS per benchmark"
P4MFE_BENCH_TIER=$tier python3 "$here/peak_rss.py" "$out/rss.json" "${benches[@]}"

python3 "$here/environment.py" "$build" > "$out/ENVIRONMENT.md"
python3 "$here/summarize.py" "$out" > "$out/SUMMARY.md"
rm -f "$out"/cpp/*.stderr
echo "Baseline written to $out"
