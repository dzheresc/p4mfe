// Workload W2 (docs/performance-testing-plan.md): the book's two normal CDF
// approximations against the erfc-based norm_cdf. norm_cdf_simpson runs an
// adaptive Simpson integral per call, so it gets 100x fewer points.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/cdf_approx.hpp>
#include <p4mfe/normal.hpp>
#include <vector>

namespace {

using p4mfe::bench::set_items;

std::vector<double> grid(std::size_t n) {
  std::vector<double> x(n);
  for (std::size_t i = 0; i < n; ++i)
    x[i] =
        -8.0 + 16.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(n);
  return x;
}

const std::vector<double>& points() {
  static const auto x =
      grid(p4mfe::bench::by_tier<std::size_t>(10'000, 1'000'000, 10'000'000));
  return x;
}
const std::vector<double>& few_points() {
  static const auto x =
      grid(p4mfe::bench::by_tier<std::size_t>(100, 10'000, 100'000));
  return x;
}

template <class F>
void sweep(benchmark::State& state, const std::vector<double>& xs, F f,
           double max_error) {
  double worst = 0.0;
  for (const double x : xs)
    worst = std::max(worst, std::abs(f(x) - p4mfe::norm_cdf(x)));
  if (!p4mfe::bench::check(state, worst < max_error,
                           "approximation error larger than expected"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const double x : xs) sum += f(x);
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, xs.size(), "points");
  state.counters["max_abs_error"] = worst;
}

const bool registered = [] {
  (void)points();
  (void)few_points();
  benchmark::RegisterBenchmark(
      "W2/ch03/norm_cdf_poly", [](benchmark::State& s) {
        sweep(
            s, points(), [](double x) { return p4mfe::ch03::norm_cdf_poly(x); },
            1e-6);
      });
  benchmark::RegisterBenchmark(
      "W2/ch03/norm_cdf_simpson", [](benchmark::State& s) {
        sweep(
            s, few_points(),
            [](double x) { return p4mfe::ch03::norm_cdf_simpson(x); }, 1e-10);
      });
  return true;
}();

}  // namespace
