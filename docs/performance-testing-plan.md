# Performance testing plan

Status: proposal. Nothing here is built yet.

## Goals

1. Measure speed and memory of every main code path in chapters 1 to 10, on
   workloads with realistic shapes and sizes (option chains, bond universes,
   yield curves, portfolios, Monte Carlo runs, PDE grids and trees).
2. **Record a baseline first.** Before any optimization lands, the current
   numbers are captured and committed to the repository, with the machine,
   OS, compiler, flags and library versions written next to them. Every later
   change is judged against that baseline.
3. Catch regressions in CI without flaky failures: gate on deterministic
   numbers (allocation counts and bytes), report timings.
4. C++ only for now; Python is evaluated later (see below).

Non-goals for now: changing any algorithm, multithreading, SIMD. Those come
after the baseline, each as its own PR with a before/after table.

## What the code looks like, from a performance point of view

Everything is header-only templates (`p4mfe_add_module` creates INTERFACE
libraries), so a benchmark translation unit instantiates exactly what it
measures and the compiler sees through the callables. Benchmarks must use
`benchmark::DoNotOptimize` / `ClobberMemory` on results, or whole loops will
be folded away.

The paths that do real work, grouped by cost profile:

| Profile | Code | Cost driver |
|---|---|---|
| Closed forms, per instrument | `ch03::bs_price`, `bs_greeks`, `ch08` barrier formulas, `ch06::bs_atm`, `ch04::rn_expected_payoff`, `common::norm_cdf/norm_inv` | `exp`, `log`, `erfc` per call; throughput over a chain |
| Scalar root finding | `ch03::implied_vol_newton/bisection`, `ch05::implied_vol`, `ch05::bisection/newton/secant`, `ch02::bond_yield`, `ch05::bond_yield` | iterations x pricing cost; depends on moneyness and starting guess |
| Quadrature | `ch02::simpson/trapezoidal/midpoint_rule`, `integrate_to_tolerance` (doubling up to 2^24), `improper_integral`, `gamma_function`, `ch03::norm_cdf_simpson`, `ch03::expected_value`, `ch04::risk_neutral_price`, `ch08::heat_solution`, `ch02::bond_price_from_inst_rate` | function evaluations; tolerance-driven loops can run to millions of points |
| 2-D quadrature | `ch08::double_integral`, `double_integral_region`, `polar_integral`, `change_of_variables_integral` | nx x ny evaluations |
| Curves and bonds | `ch02::bond_cash_flows`, `bond_metrics`, `bond_price_*`, `bootstrap_zero_rates`; `ch05::bootstrap_zero_curve`, `zero_rate_linear`; `ch06` dollar duration, immunization, yield curve shifts | cash flows per bond x bonds; allocations per bond |
| Monte Carlo | `ch04::monte_carlo_price`, `ch07::gbm_path`, `delta_hedge_stats`, `ch08::box_muller_samples` | RNG + `exp` per draw; `gbm_path` allocates a vector per path |
| Grids and trees | `ch08::solve_heat_explicit` / `european_heat_fd` (explicit, nt chosen from stability, so time is O(nx^3)), `ch08::binomial_price` (O(n^2), `std::pow` inside the American loop), `ch07::solve_bvp` / `solve_tridiagonal` (O(n)), `ch07` ODE solvers (O(n), backward Euler runs Newton per step) | grid size; memory O(n) |
| Dense linear algebra | `ch05::solve_linear` (Gaussian elimination, takes the `Matrix` by value), `newton_nd`, `approximate_newton_nd` (N+1 evaluations of F per Jacobian), `ch09::frontier_constants`, `global_min_variance_portfolio`, `lagrange_stationary_point`, `null_space` | O(N^3) per solve, N^2 doubles per matrix, one `Vector` allocation per F call |
| Appendix algorithms | `ch10::lagrange_interpolate` (O(n^2) per point), `divided_differences`, `interpolating_polynomial`, `cauchy_product`, recursions, sums | node count, series length |

Things I expect the benchmarks to show (from reading the code, not measured
yet, which is the point of the baseline):

- `ch02::bootstrap_zero_rates` looks up known discount factors by linear scan,
  so it is quadratic in the number of bonds times coupons.
- `ch02::bond_price_from_inst_rate` runs `integrate_to_tolerance` at
  `tol = 1e-12` per cash-flow interval; on a 30-year semi-annual bond that is
  60 adaptive integrations per price.
- `ch03::norm_cdf_simpson` is an adaptive integral per call; it should be
  orders of magnitude slower than `norm_cdf` (erfc).
- `ch07::delta_hedge_stats` allocates one path vector per Monte Carlo path.
- `ch05::solve_linear` copies the matrix; `newton_nd` and the FD Jacobian
  allocate a fresh `Vector` per evaluation of F.
- `ch08::binomial_price` calls `std::pow` n^2/2 times for American options.
- `ch08::european_heat_fd` grows as nx^3 because of the explicit stability
  limit.

These are hypotheses for the baseline to confirm or refute, not changes to
make now.

## Workloads and data sizes

All market data is generated in code from a fixed seed (`std::mt19937_64`,
seed 42), so runs are reproducible and no data files are checked in. The
generators live in one header, `bench/include/p4mfe/bench/market_data.hpp`.
Where a workload inverts a model (implied vol, yields, bootstrapping), the
inputs are produced from known parameters, and setup checks once that the
inversion recovers them. A benchmark that silently computes garbage fast is
worse than none.

Each workload has three size tiers, chosen at run time with
`P4MFE_BENCH_TIER=smoke|standard|large` (default `standard`):

- **smoke**: seconds in total; runs on every PR to prove the benchmarks build
  and run.
- **standard**: the baseline and the nightly numbers; about 10 to 15 minutes
  for the whole suite on a 4-core CI runner.
- **large**: memory stress, run by hand or on demand; capped at about 2 GB
  peak RSS so it fits a 16 GB GitHub runner.

### W1. Index option chain (ch01, ch03, ch04, ch05, ch06, ch07, ch08)

An SPX-like chain: spot 5000, r = 4.5%, q = 1.5%, expiries from 1 week to 3
years, strikes at moneyness 0.5 to 1.5, calls and puts. Vols from a smile with
term structure, sigma(k, T) = 0.18 + 0.10 k^2 - 0.06 k + 0.02 / sqrt(T),
k = log(K/F). Prices are Black-Scholes at that vol.

| Tier | Expiries x strikes x types | Options |
|---|---|---|
| smoke | 4 x 25 x 2 | 200 |
| standard | 24 x 200 x 2 | 9,600 |
| large | 10 underlyings x 30 x 400 x 2 | 240,000 |

Benchmarks (reported as options/second):

- price, Greeks: `bs_price`, `bs_greeks`, `rn_expected_payoff`, `bs_atm`
  approximations on the near-ATM slice;
- implied vol: `ch03::implied_vol_newton`, `ch05::implied_vol`, and
  `implied_vol_bisection`, separately for ATM, wings and 1-week expiries
  (where Newton's starting guess is worst), plus iteration counts as counters;
- bump-and-reprice: `ch07::fd_greeks` vs closed form;
- put-call parity and price bounds checks from ch01 over the chain;
- barriers: `down_and_out_call` etc. at B = 0.8 S and 1.2 S over the chain;
- PDE residual check `ch07::pde_residual` on the chain's Greeks.

### W2. Normal distribution kernels (common, ch03, ch08)

10^6 points (smoke 10^4, large 10^7) uniformly in [-8, 8] and probabilities in
(1e-12, 1 - 1e-12): `norm_cdf`, `norm_pdf`, `norm_inv`, `ch03::norm_cdf_poly`,
`ch03::norm_cdf_simpson` (10^4 points only), `ch08::box_muller` pairs. Reported
as evaluations/second. This is the cost floor under W1.

### W3. Treasury curve and bond universe (ch02, ch05, ch06)

- **Exact bootstrap** (`ch02::bootstrap_zero_rates`): semi-annual par bonds at
  every coupon date, 0.5 to 30 years, so 60 bonds (smoke 10, large 120 with
  quarterly coupons to 30 years).
- **Sparse bootstrap** (`ch05::bootstrap_zero_curve`): on-the-run style set of
  2y, 3y, 5y, 7y, 10y, 20y, 30y plus bills, about 12 bonds; and a dense set of
  60 to show Newton cost per bond.
- **Bond portfolio** pricing and risk: 10,000 corporate bonds (smoke 100, large
  1,000,000), coupons 0 to 8%, maturities 0.25 to 30 years, frequencies 1, 2,
  4, generated off the bootstrapped curve plus a random spread. Per bond:
  `bond_cash_flows`, `bond_price`, `bond_metrics`, `bond_yield`
  (both ch02 and ch05), `bond_price_from_zero_rate` against the bootstrapped
  curve, `bond_price_discrete`, `modified_duration`, `ch06` dollar duration,
  convexity, DV01 (reprice and central). Portfolio level:
  `portfolio_dollar_risk`, `immunize`, parallel and non-parallel curve shifts.
- **Instantaneous rate curve**: `bond_price_from_inst_rate` for 100 bonds
  (smoke 5) at the default tolerance, reported separately because it is
  quadrature-bound.

Bonds/second, plus allocations per bond (cash-flow vectors dominate).

### W4. Quadrature (ch02, ch03, ch04, ch08)

- Fixed-n rules on smooth, peaked and oscillatory integrands
  (`exp(-x^2)`, `1/(1 + 100 x^2)`, `sin(50 x)`), n = 2^10 to 2^22 (smoke up to
  2^12): `midpoint_rule`, `trapezoidal_rule`, `simpson_rule`; check complexity
  is O(n) with `SetComplexityN`.
- `integrate_to_tolerance` at tol 1e-6, 1e-9, 1e-12; record the accepted n as a
  counter.
- `improper_integral`, `gamma_function` for x in {1.5, 5, 20}.
- `ch04::risk_neutral_price` of a call, digital and a power payoff; versus the
  closed form.
- `ch08::double_integral` and `polar_integral` at 200x200, 1000x1000 and
  4000x4000 (large).

### W5. Monte Carlo and hedging (ch04, ch07, ch08)

- `ch04::monte_carlo_price`: 10^4, 10^6, 10^7 (large 10^8) paths, antithetic on
  and off. Memory should be O(1); the benchmark asserts it.
- `ch08::box_muller_samples`: 10^6 standard, 10^7 large (80 MB of output, a
  memory test).
- `ch07::delta_hedge_stats`: daily rebalancing (252 steps) x 10,000 paths;
  hourly (252 x 7 steps) x 1,000 paths; large: 100,000 paths. Paths/second and
  allocations per path.

### W6. PDE grids and trees (ch07, ch08)

- `ch08::european_heat_fd`: nx = 100, 200, 400, 800 (large 1600, 3200), nt
  from the stability limit. Report time, the price error against
  `bs_price`, and fit the time complexity (expected nx^3).
- `ch08::binomial_price`, European and American: n = 100, 500, 2,000, 10,000
  (large 50,000). O(n^2) time, O(n) memory. `early_exercise_premium` over 50
  strikes at n = 500.
- `ch07::solve_bvp` / `solve_tridiagonal`: n = 10^3 to 10^7.
- `ch07` ODE solvers (forward, backward Euler, leapfrog): n = 10^3 to 10^6;
  output is 2(n+1) doubles, so memory scales linearly and is checked.

### W7. Linear algebra, Newton in N dimensions, portfolios (ch05, ch08, ch09)

- `ch05::solve_linear`: N = 10, 50, 100, 250, 500 (large 1000); O(N^3).
- `newton_nd` and `approximate_newton_nd` (forward and central) on a
  nonlinear system of size N = 2, 10, 50, 100; reports iterations, F
  evaluations and allocations.
- Portfolio of N assets with a covariance matrix from a 5-factor model plus
  idiosyncratic noise (positive definite by construction): N = 10, 100, 500
  (an S&P 500 sized universe), 1000 (large). `frontier_constants`,
  `global_min_variance_portfolio`, the frontier traced at 100 target returns,
  `portfolio_variance`.
- `ch09::lagrange_stationary_point` and `classify_constrained` on the book's
  mean-variance problem for N = 3, 10, 30 (finite-difference Hessians make it
  expensive quickly).
- `ch08` chain rule, extrema and gradient/Hessian helpers: one small case each.

### W8. Appendix and calculus helpers (ch01, ch06, ch10)

- `ch10::lagrange_interpolate`, `divided_differences` + `newton_interpolate`,
  `interpolating_polynomial` with 10, 50, 200 Chebyshev nodes, evaluated at
  10^4 points.
- `cauchy_product`, `exp_power_series`, radius estimates on series of length
  100 to 10,000.
- Linear recursions, sums of powers, Stirling, Taylor convergence: one small
  case each (these are cheap; included for coverage).
- `ch06` returns on a 10-year daily price series (2,520 prices) for 500 assets:
  `log_returns`, `percentage_returns`, portfolio returns.
- `ch01` derivative helpers, forwards with discrete dividends, and the
  multivariable helpers: one case each.

### Exercises

The 120 exercise headers are mostly closed-form answers. They are not
benchmarked one by one. They get a single run of each chapter's
`exercises/examples/solutions.cpp` under the memory tools below, so leaks or
undefined behaviour still show up.

## Covering all code paths

"Every main path is exercised" becomes a checkable claim:

1. Build the benchmarks with `--coverage` (GCC) and run the **smoke** tier.
2. `gcovr` reports line and function coverage of `cpp/*/include` from the
   benchmarks alone.
3. Target: every public function in the chapter headers (not exercises) is
   called by at least one benchmark. A small script lists functions with zero
   calls; that list must be empty or explicitly waived in
   `bench/coverage-waivers.txt` with a reason.

Branch-sensitive paths get their own cases rather than one average: Newton
vs bisection fallback in implied vol, calls vs puts, European vs American,
antithetic on and off, forward vs central differences, finite vs infinite
integration limits, with and without a starting curve in bootstrapping.

## How speed is measured

[Google Benchmark](https://github.com/google/benchmark) (v1.9.x), pulled in the
same way GoogleTest is now: `find_package(benchmark CONFIG)` first, then
`FetchContent` (new `cmake/GoogleBenchmark.cmake`).

- Benchmarks are registered from code with `benchmark::RegisterBenchmark`, so
  the sizes come from the selected tier.
- Throughput counters (`SetItemsProcessed`) give options/s, bonds/s, paths/s.
- `SetComplexityN` + `Complexity()` fits the asymptotic order for the grid,
  tree, and linear algebra workloads; a wrong order is a finding by itself.
- Custom counters for iterations, function evaluations and accepted n.
- Baseline runs use `--benchmark_repetitions=10
  --benchmark_report_aggregates_only=true`, compare the median, and keep the
  coefficient of variation in the record so noise is visible.
- Local runs: `Release` build, CPU governor set to `performance`, pinned with
  `taskset -c 2`, no turbo where possible. `RelWithDebInfo` for profiling with
  `perf record` when something needs explaining.

## How memory is measured

Three levels, from cheapest and most deterministic to most detailed:

1. **Allocation counting inside the benchmark.** The benchmark `main` replaces
   global `operator new` / `operator delete` (all variants, including aligned)
   with counting versions, and registers a `benchmark::MemoryManager`. Google
   Benchmark then reports, per benchmark, number of allocations, total bytes
   allocated, and peak bytes in use during one run. These numbers are
   deterministic for a given compiler and standard library, so CI can gate on
   them.
2. **Peak RSS per workload.** Each large workload runs as its own process
   (`--benchmark_filter`), and the runner records `ru_maxrss` from
   `getrusage` (or `/usr/bin/time -v`). This catches memory the counters cannot
   see, such as the RNG state and stack.
3. **Profiles on demand.** `heaptrack` or `valgrind --tool=massif` for
   allocation call stacks when (1) or (2) moves. `valgrind memcheck` and an
   ASan + UBSan build of the smoke tier and the exercise programs, for leaks
   and undefined behaviour.

For each workload the plan writes down the expected memory model up front (for
example, heat FD: 3(nx+1) doubles; binomial: n+1 doubles; `solve_linear`: one
N x N copy; Monte Carlo price: O(1); `box_muller_samples`: n doubles), and the
benchmark checks the measured peak against it with a tolerance. A mismatch is
a bug or a surprise worth knowing about.

## Baseline (step 1, before any optimization)

The baseline is the first deliverable after the harness itself, and no
performance change is merged before it exists.

- Command: `cmake --preset bench` then `tools/bench/run_baseline.sh`, which
  runs the **standard** tier with 10 repetitions and the peak-RSS pass, and
  writes JSON.
- Stored at `bench/baselines/<date>-<machine>/`:
  - `cpp.json` (Google Benchmark JSON, includes the memory manager fields),
  - `rss.json` (peak RSS per workload),
  - `python.json` (pytest-benchmark JSON, see below),
  - `ENVIRONMENT.md`: CPU model, core count, RAM, OS and kernel, compiler and
    version, standard library, CMake build type and full flags, Google
    Benchmark version, git commit, CPU governor and
    pinning used, and anything unusual.
  - `SUMMARY.md`: a readable table per workload (median time, throughput,
    allocations, peak bytes, peak RSS).
- Two baselines are recorded: one on a GitHub `ubuntu-latest` runner (the CI
  reference, noisy but what regressions are compared against) and one on a
  quiet local machine (the reference for judging optimizations).
- Every later optimization PR includes a before/after table against the
  baseline from the same machine, produced by Google Benchmark's
  `tools/compare.py`.

## CMake integration

- New option `P4MFE_BUILD_BENCHMARKS` (default `OFF`), so normal builds and the
  current CI are unaffected.
- `p4mfe_add_module` gains a `bench/*.cpp` glob next to `tests/` and
  `examples/`, producing `p4mfe_<name>_bench` per chapter and linking a small
  `p4mfe_bench_support` library (the counting allocator, the custom `main`,
  tier selection, the market data generators).
- A `bench` custom target runs all chapter benchmarks and writes JSON to
  `build/bench/`.
- `CMakePresets.json` with `bench` (Release), `bench-profile`
  (RelWithDebInfo) and `bench-coverage` (`--coverage`, smoke tier) presets.

## CI integration

- **Every PR (added to `ci.yml`)**: configure with
  `-DP4MFE_BUILD_BENCHMARKS=ON`, build, run the **smoke** tier. Fails only if a
  benchmark fails to build, crashes, or its correctness check in setup fails.
  Adds well under a minute.
- **Nightly and manual (`bench.yml`, `schedule` + `workflow_dispatch`)**: run
  the **standard** tier, upload JSON as an artifact, compare with the
  committed CI baseline and write the comparison to the job summary.
  - Gate (fail) on allocation count and peak bytes increasing beyond 5%, since
    they are deterministic.
  - Timings are reported, not gated; GitHub-hosted runners vary by 10 to 20%.
    A slowdown over 25% on the median of 10 repetitions is flagged as a
    warning.
- **Weekly or manual**: ASan + UBSan smoke run, valgrind memcheck on the
  exercise programs, and the coverage report.
- Updating the baseline is a deliberate PR that replaces the JSON and
  `ENVIRONMENT.md`, never automatic.

## Python (deferred)

C++ comes first. The Python tree is not benchmarked in this plan. Once the
C++ baseline exists and optimizations are under way, each change will be
checked for whether it also applies to the Python mirror, and Python
benchmarks (pytest-benchmark for time, tracemalloc for memory, on the same
workloads at smaller sizes) will be added only where that is useful.

## Phases

1. **Harness** (one PR): CMake option and presets, Google Benchmark fetch,
   counting allocator and custom main, tier selection, market data generators
   with self-checks, W1 end to end, PR smoke job.
2. **Workloads** (one PR, or one per 2 to 3 chapters): W2 to W8, coverage
   check with waivers.
3. **Baseline** (one PR): baseline runs on the CI runner and the local
   machine, committed with environment notes and summaries; nightly workflow
   with comparison and allocation gate.
4. Only then: optimizations, each with a before/after table against the
   baseline, and a note on whether the change also applies to the Python
   mirror.

## Open questions

1. Google Benchmark is assumed. Fine, or would you prefer nanobench or
   Catch2's benchmarking?
2. Which local machine should hold the "quiet" baseline, and can it be pinned
   to a fixed CPU frequency?
3. Is the ~2 GB cap on the large tier right, or do you want a bigger memory
   stress run somewhere else?
