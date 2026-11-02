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
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;
namespace ch02 = p4mfe::ch02;

const std::vector<Bond>& Universe() {
  static const auto kU =
      p4mfe::bench::BondUniverse(p4mfe::bench::BondUniverseSize());
  return kU;
}

// A smaller universe for the quadrature-bound instantaneous-rate pricer.
const std::vector<Bond>& SmallUniverse() {
  static const auto kU = p4mfe::bench::BondUniverse(
      p4mfe::bench::ByTier<std::size_t>(5, 100, 1000), 7);
  return kU;
}

void BmCashFlows(benchmark::State& state) {
  const auto& u = Universe();
  for (auto _ : state) {
    std::size_t total = 0;
    for (const auto& b : u)
      total +=
          ch02::BondCashFlows(b.cf.times.back(), b.coupon, b.freq).times.size();
    benchmark::DoNotOptimize(total);
  }
  SetItems(state, u.size(), "bonds");
}

void BmPrice(benchmark::State& state) {
  const auto& u = Universe();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) sum += ch02::BondPrice(b.cf, 0.05);
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, u.size(), "bonds");
}

void BmPriceDiscrete(benchmark::State& state) {
  const auto& u = Universe();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u)
      sum += ch02::BondPriceDiscrete(b.cf, 0.05, b.freq) +
             ch02::ModifiedDuration(b.cf, 0.05, b.freq);
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, u.size(), "bonds");
}

void BmMetrics(benchmark::State& state) {
  const auto& u = Universe();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) {
      const auto m = ch02::ComputeBondMetrics(b.cf, 0.05);
      sum += m.price + m.duration + m.convexity + ch02::Dv01(b.cf, 0.05);
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, u.size(), "bonds");
}

void BmPriceFromZeroRate(benchmark::State& state) {
  const auto& u = Universe();
  double worst = 0.0;
  for (const auto& b : u) {
    const double p = ch02::BondPriceFromZeroRate(
        b.cf,
        [s = b.spread](double t) { return p4mfe::bench::TrueZeroRate(t) + s; });
    worst = std::max(worst, std::abs(p - b.price));
  }
  if (!Check(state, worst < 1e-9,
             "zero-rate pricing does not reproduce the generated prices"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u)
      sum += ch02::BondPriceFromZeroRate(b.cf, [s = b.spread](double t) {
        return p4mfe::bench::TrueZeroRate(t) + s;
      });
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, u.size(), "bonds");
}

// Adaptive Simpson per cash-flow interval at tol 1e-12, on the true curve's
// instantaneous rate.
using p4mfe::bench::TrueInstRate;

void BmPriceFromInstRate(benchmark::State& state) {
  const auto& u = SmallUniverse();
  double worst = 0.0;
  for (const auto& b : u) {
    const double expected =
        ch02::BondPriceFromZeroRate(b.cf, p4mfe::bench::TrueZeroRate);
    worst = std::max(
        worst,
        std::abs(ch02::BondPriceFromInstRate(b.cf, TrueInstRate) - expected));
  }
  if (!Check(state, worst < 1e-5,
             "instantaneous-rate pricing does not match zero-rate pricing"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u)
      sum += ch02::BondPriceFromInstRate(b.cf, TrueInstRate);
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, u.size(), "bonds");
}

void BmYield(benchmark::State& state) {
  const auto& u = Universe();
  double worst = 0.0;
  for (const auto& b : u)
    worst = std::max(
        worst, std::abs(ch02::BondPrice(b.cf, ch02::BondYield(b.cf, b.price)) -
                        b.price));
  if (!Check(state, worst < 1e-8, "bond_yield does not reprice")) return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) sum += ch02::BondYield(b.cf, b.price);
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, u.size(), "bonds");
}

// Exact bootstrap over a strip of semi-annual par bonds; range(0) is the
// number of bonds (maturity range(0)/2 years).
void BmBootstrapExact(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto quotes =
      p4mfe::bench::ParBondStrip(static_cast<double>(n) / 2.0, 2);
  const auto curve = ch02::BootstrapZeroRates(quotes, 2);
  double worst = 0.0;
  for (std::size_t i = 0; i < curve.times.size(); ++i)
    worst = std::max(
        worst,
        std::abs(curve.rates[i] - p4mfe::bench::TrueZeroRate(curve.times[i])));
  if (!Check(state, worst < 1e-10, "bootstrap does not recover the true curve"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(ch02::BootstrapZeroRates(quotes, 2));
  state.SetComplexityN(static_cast<std::int64_t>(n));
  SetItems(state, n, "bonds");
}

const bool kRegistered = [] {
  (void)Universe();
  (void)SmallUniverse();
  benchmark::RegisterBenchmark("W3/ch02/bond_cash_flows", BmCashFlows);
  benchmark::RegisterBenchmark("W3/ch02/bond_price", BmPrice);
  benchmark::RegisterBenchmark("W3/ch02/bond_price_discrete+modified_duration",
                               BmPriceDiscrete);
  benchmark::RegisterBenchmark("W3/ch02/bond_metrics+dv01", BmMetrics);
  benchmark::RegisterBenchmark("W3/ch02/bond_price_from_zero_rate",
                               BmPriceFromZeroRate);
  benchmark::RegisterBenchmark("W3/ch02/bond_price_from_inst_rate",
                               BmPriceFromInstRate);
  benchmark::RegisterBenchmark("W3/ch02/bond_yield", BmYield);
  auto* b = benchmark::RegisterBenchmark("W3/ch02/bootstrap_zero_rates",
                                         BmBootstrapExact);
  for (const auto n : p4mfe::bench::ByTier<std::vector<std::int64_t>>(
           {10, 20}, {20, 60, 120}, {60, 120, 240, 480}))
    b->Arg(n);
  b->Complexity();
  return true;
}();

}  // namespace
