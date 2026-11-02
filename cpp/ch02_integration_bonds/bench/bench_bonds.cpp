// Workload W3 (docs/performance-testing-plan.md), chapter 2 part: cash flows,
// pricing, duration and convexity, yields, and exact bootstrapping.
//
// Bonds/second for per-bond work; bootstrapping is timed per curve, with a
// complexity fit over the number of bonds.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/bonds.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch02/rates.hpp>
#include <vector>

namespace {

using p4mfe::bench::Bond;
using p4mfe::bench::check;
using p4mfe::bench::set_items;
namespace ch02 = p4mfe::ch02;

const std::vector<Bond>& universe() {
  static const auto u =
      p4mfe::bench::bond_universe(p4mfe::bench::bond_universe_size());
  return u;
}

// A smaller universe for the quadrature-bound instantaneous-rate pricer.
const std::vector<Bond>& small_universe() {
  static const auto u = p4mfe::bench::bond_universe(
      p4mfe::bench::by_tier<std::size_t>(5, 100, 1000), 7);
  return u;
}

void BM_cash_flows(benchmark::State& state) {
  const auto& u = universe();
  for (auto _ : state) {
    std::size_t total = 0;
    for (const auto& b : u)
      total += ch02::bond_cash_flows(b.cf.times.back(), b.coupon, b.freq)
                   .times.size();
    benchmark::DoNotOptimize(total);
  }
  set_items(state, u.size(), "bonds");
}

void BM_price(benchmark::State& state) {
  const auto& u = universe();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) sum += ch02::bond_price(b.cf, 0.05);
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, u.size(), "bonds");
}

void BM_price_discrete(benchmark::State& state) {
  const auto& u = universe();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u)
      sum += ch02::bond_price_discrete(b.cf, 0.05, b.freq) +
             ch02::modified_duration(b.cf, 0.05, b.freq);
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, u.size(), "bonds");
}

void BM_metrics(benchmark::State& state) {
  const auto& u = universe();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) {
      const auto m = ch02::bond_metrics(b.cf, 0.05);
      sum += m.price + m.duration + m.convexity + ch02::dv01(b.cf, 0.05);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, u.size(), "bonds");
}

void BM_price_from_zero_rate(benchmark::State& state) {
  const auto& u = universe();
  double worst = 0.0;
  for (const auto& b : u) {
    const double p =
        ch02::bond_price_from_zero_rate(b.cf, [s = b.spread](double t) {
          return p4mfe::bench::true_zero_rate(t) + s;
        });
    worst = std::max(worst, std::abs(p - b.price));
  }
  if (!check(state, worst < 1e-9,
             "zero-rate pricing does not reproduce the generated prices"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u)
      sum += ch02::bond_price_from_zero_rate(b.cf, [s = b.spread](double t) {
        return p4mfe::bench::true_zero_rate(t) + s;
      });
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, u.size(), "bonds");
}

// Adaptive Simpson per cash-flow interval at tol 1e-12, on the true curve's
// instantaneous rate.
using p4mfe::bench::true_inst_rate;

void BM_price_from_inst_rate(benchmark::State& state) {
  const auto& u = small_universe();
  double worst = 0.0;
  for (const auto& b : u) {
    const double expected =
        ch02::bond_price_from_zero_rate(b.cf, p4mfe::bench::true_zero_rate);
    worst = std::max(
        worst, std::abs(ch02::bond_price_from_inst_rate(b.cf, true_inst_rate) -
                        expected));
  }
  if (!check(state, worst < 1e-5,
             "instantaneous-rate pricing does not match zero-rate pricing"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u)
      sum += ch02::bond_price_from_inst_rate(b.cf, true_inst_rate);
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, u.size(), "bonds");
}

void BM_yield(benchmark::State& state) {
  const auto& u = universe();
  double worst = 0.0;
  for (const auto& b : u)
    worst = std::max(
        worst,
        std::abs(ch02::bond_price(b.cf, ch02::bond_yield(b.cf, b.price)) -
                 b.price));
  if (!check(state, worst < 1e-8, "bond_yield does not reprice")) return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) sum += ch02::bond_yield(b.cf, b.price);
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, u.size(), "bonds");
}

// Exact bootstrap over a strip of semi-annual par bonds; range(0) is the
// number of bonds (maturity range(0)/2 years).
void BM_bootstrap_exact(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto quotes =
      p4mfe::bench::par_bond_strip(static_cast<double>(n) / 2.0, 2);
  const auto curve = ch02::bootstrap_zero_rates(quotes, 2);
  double worst = 0.0;
  for (std::size_t i = 0; i < curve.times.size(); ++i)
    worst =
        std::max(worst, std::abs(curve.rates[i] -
                                 p4mfe::bench::true_zero_rate(curve.times[i])));
  if (!check(state, worst < 1e-10, "bootstrap does not recover the true curve"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(ch02::bootstrap_zero_rates(quotes, 2));
  state.SetComplexityN(static_cast<std::int64_t>(n));
  set_items(state, n, "bonds");
}

const bool registered = [] {
  (void)universe();
  (void)small_universe();
  benchmark::RegisterBenchmark("W3/ch02/bond_cash_flows", BM_cash_flows);
  benchmark::RegisterBenchmark("W3/ch02/bond_price", BM_price);
  benchmark::RegisterBenchmark("W3/ch02/bond_price_discrete+modified_duration",
                               BM_price_discrete);
  benchmark::RegisterBenchmark("W3/ch02/bond_metrics+dv01", BM_metrics);
  benchmark::RegisterBenchmark("W3/ch02/bond_price_from_zero_rate",
                               BM_price_from_zero_rate);
  benchmark::RegisterBenchmark("W3/ch02/bond_price_from_inst_rate",
                               BM_price_from_inst_rate);
  benchmark::RegisterBenchmark("W3/ch02/bond_yield", BM_yield);
  auto* b = benchmark::RegisterBenchmark("W3/ch02/bootstrap_zero_rates",
                                         BM_bootstrap_exact);
  for (const auto n : p4mfe::bench::by_tier<std::vector<std::int64_t>>(
           {10, 20}, {20, 60, 120}, {60, 120, 240, 480}))
    b->Arg(n);
  b->Complexity();
  return true;
}();

}  // namespace
