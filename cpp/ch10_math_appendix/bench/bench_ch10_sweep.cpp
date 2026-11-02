// Remaining chapter 10 paths (coverage sweep): big-O order estimates,
// interpolation error bounds, affine recursions, parity helpers, Stirling,
// closed-form sums and the Taylor-convergence helpers, each over a
// 1,000-point grid.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch10/big_o.hpp>
#include <p4mfe/ch10/interpolation.hpp>
#include <p4mfe/ch10/linear_recursions.hpp>
#include <p4mfe/ch10/parity.hpp>
#include <p4mfe/ch10/power_series.hpp>
#include <p4mfe/ch10/stirling.hpp>
#include <p4mfe/ch10/sums.hpp>
#include <p4mfe/ch10/taylor_convergence.hpp>
#include <vector>

namespace {

namespace ch10 = p4mfe::ch10;
using p4mfe::bench::check;
using p4mfe::bench::set_items;

constexpr std::size_t n_points = 1000;

std::vector<double> grid(double lo, double hi) {
  std::vector<double> x(n_points);
  for (std::size_t i = 0; i < n_points; ++i)
    x[i] = lo + (hi - lo) * static_cast<double>(i) /
                    static_cast<double>(n_points - 1);
  return x;
}

void BM_orders_and_bounds(benchmark::State& state) {
  const auto xs = grid(0.01, 1.0);
  const std::vector<double> nodes{0.0, 0.25, 0.5, 0.75, 1.0};
  const auto cube = [](double h) { return h * h * h + h * h * h * h; };
  if (!check(state, std::abs(ch10::observed_power(cube, 1e-4) - 3.0) < 1e-3,
             "observed power is not 3"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const double x : xs) {
      sum += ch10::observed_power(cube, x) +
             ch10::observed_power_at_infinity(cube, 1.0 / x);
      sum += ch10::interpolation_error_bound(1.0, nodes, x);
      sum += ch10::first_order_affine(0.95, 0.05 * x, 1.0, 50);
    }
    auto r = ch10::order_ratios([](double x) { return std::sin(x); },
                                [](double x) { return x; }, xs);
    benchmark::DoNotOptimize(sum);
    benchmark::DoNotOptimize(r);
  }
  set_items(state, n_points, "points");
}

void BM_parity(benchmark::State& state) {
  const auto xs = grid(0.0, 3.0);
  const auto cosh_like = [](double x) { return std::exp(x) + std::exp(-x); };
  const auto mixed = [](double x) { return std::exp(x); };
  if (!check(state,
             ch10::parity(cosh_like, xs) == ch10::Parity::Even &&
                 ch10::parity([](double x) { return x * x * x; }, xs) ==
                     ch10::Parity::Odd &&
                 ch10::parity(mixed, xs) == ch10::Parity::Neither,
             "parity misclassified"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    sum += ch10::is_even(cosh_like, xs) + ch10::is_odd(mixed, xs);
    sum += static_cast<double>(ch10::parity(mixed, xs));
    for (const double x : xs)
      sum += ch10::even_part(mixed, x) + ch10::odd_part(mixed, x);
    sum += ch10::symmetric_integral(mixed, 2.0);
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, n_points, "points");
}

void BM_stirling_and_sums(benchmark::State& state) {
  const auto xs = grid(-0.95, 0.95);
  if (!check(state,
             ch10::sum_k3(1000) == ch10::sum_k(1000) * ch10::sum_k(1000) &&
                 std::abs(ch10::sum_k_xk(50, 0.5) -
                          ch10::sum_kj_xk(50, 1, 0.5)) < 1e-12,
             "closed-form sums disagree"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 1; i <= n_points; ++i) {
      const double n = static_cast<double>(i);
      const auto [lo, hi] = ch10::stirling_bounds(n);
      sum += ch10::stirling(n) / std::exp(ch10::log_factorial(n)) + hi - lo;
      sum += std::log(ch10::central_binomial(std::min(n, 500.0)));
      sum += static_cast<double>(ch10::sum_k(i) + ch10::sum_k2(i) +
                                 ch10::sum_k3(i % 1000));
    }
    for (const double x : xs) {
      sum += ch10::geometric_sum(100, x) + ch10::sum_k_xk(100, x) +
             ch10::sum_k2_xk(100, x);
      sum += ch10::sum_kj_xk(100, 3, x);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, n_points, "points");
}

void BM_taylor_convergence(benchmark::State& state) {
  const auto xs = grid(-2.0, 2.0);
  const std::vector<double> derivs(20, 1.0);  // e^x at 0
  if (!check(state,
             std::abs(ch10::taylor_series(derivs, 0.0)(1.0) - std::exp(1.0)) <
                 1e-12,
             "taylor_series is off"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    const auto s = ch10::taylor_series(derivs, 0.0);
    const auto g = ch10::geometric_power_series(50);
    const auto d = ch10::derivatives_at_center(s);
    for (const double x : xs) {
      sum += s(x) + g(0.4 * x) + ch10::flat_function(x);
      sum += ch10::taylor_bound_geometric(1.0, 1.0, 0.0, x, 10);
      sum += ch10::taylor_terms_needed(1.0, 1.0, 0.0, x, 1e-12);
    }
    benchmark::DoNotOptimize(sum + d.back());
  }
  set_items(state, n_points, "points");
}

const bool registered = [] {
  benchmark::RegisterBenchmark("W8/ch10/orders+bounds", BM_orders_and_bounds)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W8/ch10/parity", BM_parity)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W8/ch10/stirling+sums", BM_stirling_and_sums)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W8/ch10/taylor_convergence",
                               BM_taylor_convergence)
      ->Unit(benchmark::kMicrosecond);
  return true;
}();

}  // namespace
