// Remaining chapter 2 paths (coverage sweep): rate conversions, forward rates
// and discount factors on the true curve (closed form and from the
// instantaneous rate by quadrature), zero coupon bonds, duration-based price
// approximations, observed convergence order, and differentiating an
// improper integral.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/bonds.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch02/rates.hpp>
#include <vector>

namespace {

namespace ch02 = p4mfe::ch02;
using p4mfe::bench::check;
using p4mfe::bench::set_items;

// Monthly grid out to 30 years.
std::vector<double> grid() {
  std::vector<double> t;
  for (int m = 1; m <= 360; ++m) t.push_back(m / 12.0);
  return t;
}

void BM_curve_closed_forms(benchmark::State& state) {
  const auto ts = grid();
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 1; i < ts.size(); ++i) {
      const double t1 = ts[i - 1], t2 = ts[i];
      const double r1 = p4mfe::bench::true_zero_rate(t1),
                   r2 = p4mfe::bench::true_zero_rate(t2);
      const double D1 = ch02::discount_factor(r1, t1),
                   D2 = ch02::discount_factor(r2, t2);
      sum += ch02::forward_rate(r1, t1, r2, t2) +
             ch02::forward_rate_from_discounts(D1, t1, D2, t2);
      sum += ch02::discrete_from_continuous(r2, 2.0) +
             ch02::continuous_from_discrete(r2, 2.0);
      sum += ch02::inst_rate_from_zero_rate(p4mfe::bench::true_zero_rate, t2);
      const double z = ch02::zero_coupon_bond_price(t2, r2);
      sum += ch02::zero_rate_from_zero_coupon_price(z, t2);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, ts.size() - 1, "points");
}

// Curve quantities from the instantaneous rate: one adaptive integral each.
void BM_curve_from_inst_rate(benchmark::State& state) {
  std::vector<double> ts;
  for (int y = 1; y <= 30; ++y) ts.push_back(y);
  const double err = std::abs(
      ch02::zero_rate_from_inst_rate(p4mfe::bench::true_inst_rate, 10.0) -
      p4mfe::bench::true_zero_rate(10.0));
  if (!check(state, err < 1e-9, "zero rate from instantaneous rate is off"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const double t : ts) {
      sum +=
          ch02::discount_factor_from_inst_rate(p4mfe::bench::true_inst_rate, t);
      sum += ch02::zero_rate_from_inst_rate(p4mfe::bench::true_inst_rate, t);
      sum += ch02::forward_rate_from_inst_rate(p4mfe::bench::true_inst_rate,
                                               t - 1.0, t);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, ts.size(), "maturities");
}

void BM_duration_approximations(benchmark::State& state) {
  const auto u = p4mfe::bench::bond_universe(
      p4mfe::bench::by_tier<std::size_t>(100, 10'000, 100'000));
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) {
      const double B = ch02::bond_price(b.cf, 0.05);
      const double D = ch02::bond_duration(b.cf, 0.05),
                   C = ch02::bond_convexity(b.cf, 0.05);
      sum += ch02::price_approx_duration(B, D, 0.001) +
             ch02::price_approx_duration_convexity(B, D, C, 0.001);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, u.size(), "bonds");
}

void BM_integration_misc(benchmark::State& state) {
  // d/dt integral_0^inf e^(-t x^2) dx = -sqrt(pi) / (4 t^1.5).
  const auto f = [](double x, double t) { return std::exp(-t * x * x); };
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        ch02::improper_integral_derivative(f, 0.0, ch02::infinity, 1.0));
    benchmark::DoNotOptimize(
        ch02::observed_order([](double x) { return std::exp(x); }, 0.0, 1.0,
                             std::exp(1.0) - 1.0, 64, ch02::Rule::Simpson));
  }
}

const bool registered = [] {
  benchmark::RegisterBenchmark("W3/ch02/curve_closed_forms",
                               BM_curve_closed_forms);
  benchmark::RegisterBenchmark("W3/ch02/curve_from_inst_rate",
                               BM_curve_from_inst_rate)
      ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("W3/ch02/duration_approximations",
                               BM_duration_approximations);
  benchmark::RegisterBenchmark(
      "W4/ch02/improper_integral_derivative+observed_order",
      BM_integration_misc)
      ->Unit(benchmark::kMillisecond);
  return true;
}();

}  // namespace
