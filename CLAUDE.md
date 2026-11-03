# Notes for Claude

## Code style

Write all new and changed code to the Google style guides:
[C++](https://google.github.io/styleguide/cppguide.html) and
[Python](https://google.github.io/styleguide/pyguide.html). Existing code is
restyled in separate changes; do not mix a restyle into a functional change.

Formatting is enforced by the `format` CI job; run before committing:

```sh
git ls-files 'cpp/*.hpp' 'cpp/*.cpp' | xargs clang-format -i   # .clang-format, clang-format 18
pyink --config python/pyproject.toml python tools               # [tool.pyink] in python/pyproject.toml
```

Naming is enforced by the `naming` CI job (C++: `.clang-tidy`,
`readability-identifier-naming` with Google rules; Python: ruff's pep8-naming
rules in `[tool.ruff.lint]`). Check locally with:

```sh
cmake -S . -B build/tidy -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DP4MFE_BUILD_BENCHMARKS=ON -DCMAKE_CXX_COMPILER=clang++
run-clang-tidy-18 -p build/tidy -quiet "$PWD/cpp/"            # clang-tidy 18
ruff check --config python/pyproject.toml python tools
```

Book symbols (S, K, T, sigma_P, ...) stay in comments; in code use descriptive
lower_case names (`spot`, `strike`, `maturity`), and give a function a verb
(`ComputeX`, `MakeX`, `GetX`) when its CamelCase name would equal a type name.

## Performance work (C++)

The plan is [docs/performance-testing-plan.md](docs/performance-testing-plan.md).
No optimization lands without a before/after comparison against a baseline
recorded on the same machine.

### Benchmarks and baselines

```sh
cmake --preset bench && cmake --build --preset bench -j        # Release, build/bench-release
P4MFE_BENCH_TIER=smoke cmake --build --preset bench --target bench   # quick run of everything
tools/bench/run_baseline.sh <out-dir> [reps]    # timing (10 reps), heap, peak RSS, ENVIRONMENT.md, SUMMARY.md
tools/bench/compare.py <baseline-dir> <run-dir> # Markdown table; exit 1 if heap use grows > 5%
tools/bench/coverage.sh                         # needs gcovr; every public header function must run
```

- Tiers: `P4MFE_BENCH_TIER=smoke|standard|large` (default standard).
- Run a subset with Google Benchmark's own filter, e.g.
  `BENCHMARK_FILTER='binomial_price' tools/bench/run_baseline.sh <out-dir>`
  or `--benchmark_filter=...` on one `p4mfe_<chapter>_bench` executable.
- Committed baselines: `bench/baselines/ci` (GitHub runner, used by the
  manually run `bench` workflow) and `bench/baselines/<date>-<machine>` (local).
  Compare only with the baseline from the same machine and compiler.
- For a before/after on one machine, build `main` in a git worktree and this
  branch side by side, run both with the same filter, and paste
  `compare.py`'s table into the PR. Note whether the change also applies to
  the Python mirror.

### Profiling with the Valgrind suite

Profile the `bench-profile` preset (RelWithDebInfo, frame pointers), one
benchmark at a time, with a fixed iteration count so runs are repeatable:

```sh
cmake --preset bench-profile && cmake --build --preset bench-profile -j
E=build/bench-profile/cpp/<chapter>/p4mfe_<chapter>_bench
export P4MFE_BENCH_TIER=smoke BENCHMARK_MIN_TIME=1x
F='--benchmark_filter=^W6/ch08/binomial_price/put/american/500$'
```

| Question | Tool | Command | Read with |
|---|---|---|---|
| Leaks, invalid reads/writes, uninitialised values | memcheck | `valgrind --tool=memcheck --leak-check=full --error-exitcode=1 $E $F` | the log |
| Where instructions go (call graph, inclusive cost) | callgrind | `valgrind --tool=callgrind --callgrind-out-file=cg.out $E $F` | `callgrind_annotate --inclusive=yes cg.out`, or KCachegrind |
| Cache and branch behaviour | cachegrind | `valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes --cachegrind-out-file=cache.out $E $F` | `cg_annotate cache.out` |
| Heap over time, who holds the peak | massif | `valgrind --tool=massif --massif-out-file=ms.out $E $F` | `ms_print ms.out` |
| Heap churn (short-lived allocations) | dhat | `valgrind --tool=dhat --dhat-out-file=dhat.out $E $F` | `dh_view.html` |

Notes:

- Each executable builds its market data at startup; to see only the
  benchmark in callgrind, add `--toggle-collect='p4mfe::chNN::<function>*'`
  (or `--collect-atstart=no` plus `--toggle-collect`).
- Valgrind runs 20 to 50 times slower: use the smoke tier and `1x`.
- Valgrind does not model the cost of subnormal floating-point operations
  or real CPU frequency; confirm any hotspot with the Release benchmark before
  and after the change.
- The benchmarks replace global `operator new`/`delete` (the counting
  allocator); memcheck, massif and dhat still see every allocation because it
  forwards to `malloc`/`free`.
- Also useful: the exercise programs under memcheck, and the `bench-coverage`
  preset for line coverage (`tools/bench/coverage.sh`).
