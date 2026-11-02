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
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;

constexpr std::size_t kNPoints = 1000;

std::vector<double> Grid(double lo, double hi) {
  std::vector<double> x(kNPoints);
  for (std::size_t i = 0; i < kNPoints; ++i)
    x[i] = lo + (hi - lo) * static_cast<double>(i) /
                    static_cast<double>(kNPoints - 1);
  return x;
}

void BmOrdersAndBounds(benchmark::State& state) {
  const auto xs = Grid(0.01, 1.0);
  const std::vector<double> nodes{0.0, 0.25, 0.5, 0.75, 1.0};
  const auto cube = [](double h) { return h * h * h + h * h * h * h; };
  if (!Check(state, std::abs(ch10::ObservedPower(cube, 1e-4) - 3.0) < 1e-3,
             "observed power is not 3"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const double x : xs) {
      sum += ch10::ObservedPower(cube, x) +
             ch10::ObservedPowerAtInfinity(cube, 1.0 / x);
      sum += ch10::InterpolationErrorBound(1.0, nodes, x);
      sum += ch10::FirstOrderAffine(0.95, 0.05 * x, 1.0, 50);
    }
    auto r = ch10::OrderRatios([](double x) { return std::sin(x); },
                               [](double x) { return x; }, xs);
    benchmark::DoNotOptimize(sum);
    benchmark::DoNotOptimize(r);
  }
  SetItems(state, kNPoints, "points");
}

void BmParity(benchmark::State& state) {
  const auto xs = Grid(0.0, 3.0);
  const auto cosh_like = [](double x) { return std::exp(x) + std::exp(-x); };
  const auto mixed = [](double x) { return std::exp(x); };
  if (!Check(state,
             ch10::ClassifyParity(cosh_like, xs) == ch10::Parity::kEven &&
                 ch10::ClassifyParity([](double x) { return x * x * x; }, xs) ==
                     ch10::Parity::kOdd &&
                 ch10::ClassifyParity(mixed, xs) == ch10::Parity::kNeither,
             "parity misclassified"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    sum += ch10::IsEven(cosh_like, xs) + ch10::IsOdd(mixed, xs);
    sum += static_cast<double>(ch10::ClassifyParity(mixed, xs));
    for (const double x : xs)
      sum += ch10::EvenPart(mixed, x) + ch10::OddPart(mixed, x);
    sum += ch10::SymmetricIntegral(mixed, 2.0);
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, kNPoints, "points");
}

void BmStirlingAndSums(benchmark::State& state) {
  const auto xs = Grid(-0.95, 0.95);
  if (!Check(state,
             ch10::SumK3(1000) == ch10::SumK(1000) * ch10::SumK(1000) &&
                 std::abs(ch10::SumKXk(50, 0.5) - ch10::SumKjXk(50, 1, 0.5)) <
                     1e-12,
             "closed-form sums disagree"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 1; i <= kNPoints; ++i) {
      const double n = static_cast<double>(i);
      const auto [lo, hi] = ch10::StirlingBounds(n);
      sum += ch10::Stirling(n) / std::exp(ch10::LogFactorial(n)) + hi - lo;
      sum += std::log(ch10::CentralBinomial(std::min(n, 500.0)));
      sum += static_cast<double>(ch10::SumK(i) + ch10::SumK2(i) +
                                 ch10::SumK3(i % 1000));
    }
    for (const double x : xs) {
      sum += ch10::GeometricSum(100, x) + ch10::SumKXk(100, x) +
             ch10::SumK2Xk(100, x);
      sum += ch10::SumKjXk(100, 3, x);
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, kNPoints, "points");
}

void BmTaylorConvergence(benchmark::State& state) {
  const auto xs = Grid(-2.0, 2.0);
  const std::vector<double> derivs(20, 1.0);  // e^x at 0
  if (!Check(state,
             std::abs(ch10::TaylorSeries(derivs, 0.0)(1.0) - std::exp(1.0)) <
                 1e-12,
             "taylor_series is off"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    const auto s = ch10::TaylorSeries(derivs, 0.0);
    const auto g = ch10::GeometricPowerSeries(50);
    const auto d = ch10::DerivativesAtCenter(s);
    for (const double x : xs) {
      sum += s(x) + g(0.4 * x) + ch10::FlatFunction(x);
      sum += ch10::TaylorBoundGeometric(1.0, 1.0, 0.0, x, 10);
      sum += ch10::TaylorTermsNeeded(1.0, 1.0, 0.0, x, 1e-12);
    }
    benchmark::DoNotOptimize(sum + d.back());
  }
  SetItems(state, kNPoints, "points");
}

const bool kRegistered = [] {
  benchmark::RegisterBenchmark("W8/ch10/orders+bounds", BmOrdersAndBounds)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W8/ch10/parity", BmParity)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W8/ch10/stirling+sums", BmStirlingAndSums)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W8/ch10/taylor_convergence",
                               BmTaylorConvergence)
      ->Unit(benchmark::kMicrosecond);
  return true;
}();

}  // namespace
