// Workload W7 (docs/performance-testing-plan.md), chapter 9 part: mean-variance
// portfolios over an N-asset factor-model universe (N up to an S&P 500 sized
// universe), and the general Lagrange multiplier solver on the same problem.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/portfolio.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch09/lagrange.hpp>
#include <p4mfe/ch09/portfolio.hpp>
#include <vector>

namespace {

namespace ch09 = p4mfe::ch09;
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;

void BmFrontier(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto u = p4mfe::bench::MakeAssetUniverse(n);
  const auto gmv = ch09::GlobalMinVariancePortfolio(u.covariance);
  double sum_w = 0.0;
  for (const double w : gmv) sum_w += w;
  if (!Check(state, std::abs(sum_w - 1.0) < 1e-9,
             "minimum variance weights do not sum to 1"))
    return;
  for (auto _ : state) {
    auto c = ch09::ComputeFrontierConstants(u.mu, u.covariance);
    benchmark::DoNotOptimize(c);
    benchmark::DoNotOptimize(ch09::GlobalMinVariancePortfolio(u.covariance));
    benchmark::DoNotOptimize(ch09::TangencyPortfolio(u.mu, u.covariance, 0.03));
  }
  state.SetComplexityN(static_cast<std::int64_t>(n));
}

// The efficient frontier traced at 100 target returns: each point solves the
// covariance system again, as the API does.
void BmFrontierTrace(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto u = p4mfe::bench::MakeAssetUniverse(n);
  for (auto _ : state) {
    double sum = 0.0;
    for (int k = 0; k < 100; ++k) {
      const double target = 0.03 + 0.001 * k;
      const auto w = ch09::MinVariancePortfolio(u.mu, u.covariance, target);
      sum += ch09::PortfolioVariance(w, u.covariance);
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, 100, "points");
  state.SetComplexityN(static_cast<std::int64_t>(n));
}

void BmPortfolioVariance(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto u = p4mfe::bench::MakeAssetUniverse(n);
  const std::vector<double> w(n, 1.0 / static_cast<double>(n));
  for (auto _ : state)
    benchmark::DoNotOptimize(ch09::PortfolioVariance(w, u.covariance));
  state.SetComplexityN(static_cast<std::int64_t>(n));
}

// min w' Sigma w subject to sum w = 1 and mu' w = target, by the general
// solver (finite-difference gradients and Hessians), against the closed form.
void BmLagrange(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto u = p4mfe::bench::MakeAssetUniverse(n);
  const double target = 0.05;
  const auto f = [&](const ch09::Vector& w) {
    return ch09::PortfolioVariance(w, u.covariance);
  };
  const auto g = [&](const ch09::Vector& w) {
    double s = -1.0;
    for (const double x : w) s += x;
    return ch09::Vector{s, ch09::Dot(u.mu, w) - target};
  };
  const ch09::Vector w0(n, 1.0 / static_cast<double>(n));
  const auto res = ch09::LagrangeStationaryPoint(f, g, w0);
  const auto exact = ch09::MinVariancePortfolio(u.mu, u.covariance, target);
  double err = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    err = std::max(err, std::abs(res.x[i] - exact[i]));
  if (!Check(state, err < 1e-4,
             "Lagrange solver disagrees with the closed form"))
    return;
  for (auto _ : state) {
    const auto r = ch09::LagrangeStationaryPoint(f, g, w0);
    benchmark::DoNotOptimize(ch09::ClassifyConstrained(f, g, r.x, r.lambda));
  }
  state.counters["iterations"] = static_cast<double>(res.iterations);
  state.SetComplexityN(static_cast<std::int64_t>(n));
}

const bool kRegistered = [] {
  const auto ns = p4mfe::bench::ByTier<std::vector<std::int64_t>>(
      {10, 50}, {10, 100, 500}, {100, 500, 1000});
  auto* f =
      benchmark::RegisterBenchmark("W7/ch09/frontier+gmv+tangency", BmFrontier);
  auto* t = benchmark::RegisterBenchmark("W7/ch09/frontier_trace_100",
                                         BmFrontierTrace);
  auto* v = benchmark::RegisterBenchmark("W7/ch09/portfolio_variance",
                                         BmPortfolioVariance);
  for (const auto n : ns) {
    f->Arg(n);
    t->Arg(n);
    v->Arg(n);
  }
  f->Complexity(benchmark::oNCubed)->Unit(benchmark::kMillisecond);
  t->Complexity(benchmark::oNCubed)->Unit(benchmark::kMillisecond);
  v->Complexity(benchmark::oNSquared)->Unit(benchmark::kMicrosecond);
  auto* l = benchmark::RegisterBenchmark("W7/ch09/lagrange_stationary_point",
                                         BmLagrange);
  for (const auto n : p4mfe::bench::ByTier<std::vector<std::int64_t>>(
           {3}, {3, 10, 30}, {10, 30, 60}))
    l->Arg(n);
  l->Unit(benchmark::kMillisecond);
  return true;
}();

}  // namespace
