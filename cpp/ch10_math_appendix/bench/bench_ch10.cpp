// Workload W8 (docs/performance-testing-plan.md), chapter 10 part:
// polynomial interpolation on Chebyshev nodes (Lagrange form is O(n^2) per
// point, Newton form O(n) after O(n^2) setup), power series algebra, and the
// cheap appendix helpers for coverage.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch10/interpolation.hpp>
#include <p4mfe/ch10/linear_recursions.hpp>
#include <p4mfe/ch10/power_series.hpp>
#include <p4mfe/ch10/stirling.hpp>
#include <p4mfe/ch10/sums.hpp>
#include <vector>

namespace {

namespace ch10 = p4mfe::ch10;
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;

struct Nodes {
  std::vector<double> xs, ys, eval;
};

// Chebyshev nodes on [-1, 1] for f(x) = 1 / (1 + 25 x^2) (Runge's function),
// evaluated at 10,000 points.
Nodes Chebyshev(std::size_t n) {
  Nodes s;
  for (std::size_t i = 0; i < n; ++i) {
    const double x =
        std::cos((2.0 * static_cast<double>(i) + 1.0) * std::numbers::pi /
                 (2.0 * static_cast<double>(n)));
    s.xs.push_back(x);
    s.ys.push_back(1.0 / (1.0 + 25.0 * x * x));
  }
  const std::size_t m =
      p4mfe::bench::ByTier<std::size_t>(1'000, 10'000, 100'000);
  for (std::size_t i = 0; i < m; ++i)
    s.eval.push_back(-1.0 + 2.0 * (static_cast<double>(i) + 0.5) /
                                static_cast<double>(m));
  return s;
}

void BmLagrangeInterpolate(benchmark::State& state) {
  const auto s = Chebyshev(static_cast<std::size_t>(state.range(0)));
  for (auto _ : state) {
    double sum = 0.0;
    for (const double x : s.eval)
      sum += ch10::LagrangeInterpolate(s.xs, s.ys, x);
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, s.eval.size(), "points");
  state.SetComplexityN(state.range(0));
}

void BmNewtonInterpolate(benchmark::State& state) {
  const auto s = Chebyshev(static_cast<std::size_t>(state.range(0)));
  const auto c = ch10::DividedDifferences(s.xs, s.ys);
  double worst = 0.0;
  for (const double x : s.eval)
    worst = std::max(worst, std::abs(ch10::NewtonInterpolate(s.xs, c, x) -
                                     ch10::LagrangeInterpolate(s.xs, s.ys, x)));
  // Divided differences lose accuracy as n grows (Chebyshev nodes in their
  // natural order); the gap to the Lagrange form is reported, and only a
  // gross disagreement fails.
  if (!Check(state, worst < 1e-2, "Newton and Lagrange forms disagree")) return;
  for (auto _ : state) {
    const auto cc = ch10::DividedDifferences(s.xs, s.ys);
    double sum = 0.0;
    for (const double x : s.eval) sum += ch10::NewtonInterpolate(s.xs, cc, x);
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, s.eval.size(), "points");
  state.counters["max_gap_to_lagrange"] = worst;
}

void BmInterpolatingPolynomial(benchmark::State& state) {
  const auto s = Chebyshev(static_cast<std::size_t>(state.range(0)));
  for (auto _ : state) {
    const auto a = ch10::InterpolatingPolynomial(s.xs, s.ys);
    benchmark::DoNotOptimize(ch10::PolynomialValue(a, 0.3));
  }
  state.SetComplexityN(state.range(0));
}

void BmPowerSeries(benchmark::State& state) {
  const auto n = static_cast<unsigned>(state.range(0));
  for (auto _ : state) {
    const auto e = ch10::ExpPowerSeries(n);
    const auto l = ch10::Log1pPowerSeries(n);
    const auto p = ch10::CauchyProduct(e, l);
    const auto d = ch10::Derivative(p);
    auto a = ch10::Antiderivative(d);
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(ch10::RadiusRatioEstimate(l.a));
    benchmark::DoNotOptimize(ch10::RadiusRootEstimate(l.a));
  }
  state.SetComplexityN(state.range(0));
}

void BmAppendixHelpers(benchmark::State& state) {
  const std::vector<double> a{1.0, 1.0}, init{0.0, 1.0};
  for (auto _ : state) {
    benchmark::DoNotOptimize(ch10::RecursionTerms(a, init, 90));
    benchmark::DoNotOptimize(ch10::SolveSecondOrder(1.0, 1.0, 0.0, 1.0));
    benchmark::DoNotOptimize(ch10::LoanPayment(300'000.0, 0.06 / 12.0, 360));
    benchmark::DoNotOptimize(ch10::SumsOfPowersRecursive(1000, 6));
    benchmark::DoNotOptimize(ch10::SumOfPowersPolynomial(8));
    benchmark::DoNotOptimize(ch10::SumKjXkInfinite(4, 0.5));
    benchmark::DoNotOptimize(ch10::LogStirling(1000.0, 4));
    benchmark::DoNotOptimize(ch10::CentralBinomialStirling(500.0));
  }
}

const bool kRegistered = [] {
  const auto ns = p4mfe::bench::ByTier<std::vector<std::int64_t>>(
      {10, 50}, {10, 50, 200}, {50, 200, 1000});
  auto* l = benchmark::RegisterBenchmark("W8/ch10/lagrange_interpolate",
                                         BmLagrangeInterpolate);
  auto* n = benchmark::RegisterBenchmark(
      "W8/ch10/divided_differences+newton_interpolate", BmNewtonInterpolate);
  auto* p = benchmark::RegisterBenchmark("W8/ch10/interpolating_polynomial",
                                         BmInterpolatingPolynomial);
  for (const auto k : ns) {
    l->Arg(k);
    p->Arg(k);
  }
  // The Newton form with nodes in natural order is unusable beyond ~50
  // Chebyshev nodes (gap to the Lagrange form > 1e-2 at 200), so it stops
  // at 50.
  for (const auto k : {10, 50}) n->Arg(k);
  l->Complexity(benchmark::oNSquared)->Unit(benchmark::kMillisecond);
  n->Unit(benchmark::kMillisecond);
  p->Complexity(benchmark::oNSquared)->Unit(benchmark::kMicrosecond);
  auto* ps =
      benchmark::RegisterBenchmark("W8/ch10/power_series", BmPowerSeries);
  for (const auto k : p4mfe::bench::ByTier<std::vector<std::int64_t>>(
           {100}, {100, 1'000, 10'000}, {1'000, 10'000, 30'000}))
    ps->Arg(k);
  ps->Complexity(benchmark::oNSquared)->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W8/ch10/appendix_helpers", BmAppendixHelpers)
      ->Unit(benchmark::kMicrosecond);
  return true;
}();

}  // namespace
