// Chapter 8 benchmarks:
//   W1  barrier option closed forms over the chain (barriers at 0.8 S, 1.2 S)
//   W2  Box-Muller transform, pairwise
//   W4  double integrals (rectangle and polar) at growing grid sizes
//   W5  box_muller_samples: output is n doubles, a memory test
//   W6  heat-equation finite differences, explicit (time ~ nx^3) and
//       Crank-Nicolson (time ~ nx^2), and binomial trees (time ~ n^2,
//       memory ~ n), with accuracy against Black-Scholes
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch08/barrier.hpp>
#include <p4mfe/ch08/box_muller.hpp>
#include <p4mfe/ch08/double_integrals.hpp>
#include <p4mfe/ch08/early_exercise.hpp>
#include <p4mfe/ch08/heat_equation.hpp>
#include <string>
#include <vector>

namespace {

namespace ch08 = p4mfe::ch08;
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;
using p4mfe::ch01::OptionType;

// ----------------------------------------------------------------- W1 -----

void BmBarriers(benchmark::State& state) {
  const auto chain =
      p4mfe::bench::MakeOptionChain(p4mfe::bench::ChainSpecForTier());
  const p4mfe::bench::ChainSpec spec;
  // In + out = vanilla, checked once.
  double worst = 0.0;
  for (const auto& o : chain) {
    const double spot = o.spot, strike = o.strike, maturity = o.maturity,
                 v = o.sigma;
    const double call =
        p4mfe::ch03::BsCall(spot, strike, maturity, v, spec.r, spec.q);
    worst = std::max(
        worst, std::abs(ch08::DownAndOutCall(spot, strike, 0.8 * spot, maturity,
                                             v, spec.r, spec.q) +
                        ch08::DownAndInCall(spot, strike, 0.8 * spot, maturity,
                                            v, spec.r, spec.q) -
                        call));
  }
  if (!Check(state, worst < 1e-8 * spec.spot, "in + out != vanilla")) return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : chain) {
      const double spot = o.spot, strike = o.strike, maturity = o.maturity,
                   v = o.sigma;
      sum += ch08::DownAndOutCall(spot, strike, 0.8 * spot, maturity, v, spec.r,
                                  spec.q);
      sum += ch08::DownAndInCall(spot, strike, 0.8 * spot, maturity, v, spec.r,
                                 spec.q);
      sum += ch08::UpAndOutPut(spot, strike, 1.2 * spot, maturity, v, spec.r,
                               spec.q);
      sum += ch08::UpAndInPut(spot, strike, 1.2 * spot, maturity, v, spec.r,
                              spec.q);
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, chain.size(), "options");
}

// ----------------------------------------------------------------- W2 -----

void BmBoxMullerPairs(benchmark::State& state) {
  const std::size_t n =
      p4mfe::bench::ByTier<std::size_t>(10'000, 1'000'000, 10'000'000);
  std::vector<double> u(n + 1);
  for (std::size_t i = 0; i <= n; ++i)
    u[i] = (static_cast<double>(i) + 0.5) / static_cast<double>(n + 1);
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const auto [z1, z2] = ch08::BoxMuller(u[i], u[n - i]);
      sum += z1 + z2;
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, n, "pairs");
}

// ----------------------------------------------------------------- W4 -----

void BmDoubleIntegral(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  // integral over [0,1]^2 of exp(-(x^2 + y^2)) = (sqrt(pi)/2 erf(1))^2.
  const auto f = [](double x, double y) { return std::exp(-(x * x + y * y)); };
  const double e = std::sqrt(std::numbers::pi) / 2.0 * std::erf(1.0);
  if (!Check(state,
             std::abs(ch08::DoubleIntegral(f, 0.0, 1.0, 0.0, 1.0, n, n) -
                      e * e) < 1e-8,
             "double_integral is off"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(ch08::DoubleIntegral(f, 0.0, 1.0, 0.0, 1.0, n, n));
  state.SetComplexityN(static_cast<std::int64_t>(n * n));
  SetItems(state, n * n, "cells");
}

void BmPolarIntegral(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  // e^(-(x^2+y^2)/2) over the disk of radius 2, against the closed form.
  const auto f = [](double x, double y) {
    return std::exp(-0.5 * (x * x + y * y));
  };
  const double exact = ch08::GaussianDiskIntegral(2.0);
  if (!Check(state,
             std::abs(ch08::PolarIntegral(f, 0.0, 2.0, 0.0,
                                          2.0 * std::numbers::pi, n, n) -
                      exact) < 1e-6,
             "polar_integral is off"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch08::PolarIntegral(f, 0.0, 2.0, 0.0, 2.0 * std::numbers::pi, n, n));
  SetItems(state, n * n, "cells");
}

// ----------------------------------------------------------------- W5 -----

void BmBoxMullerSamples(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  for (auto _ : state) benchmark::DoNotOptimize(ch08::BoxMullerSamples(n));
  state.SetComplexityN(static_cast<std::int64_t>(n));
  SetItems(state, n, "samples");
}

// ----------------------------------------------------------------- W6 -----

constexpr double kS = 100.0, kK = 105.0, kT = 1.0, kSigma = 0.25, kR = 0.045,
                 kQ = 0.015;

void BmHeatCrankNicolson(benchmark::State& state, OptionType type) {
  const auto nx = static_cast<std::size_t>(state.range(0));
  const double exact = p4mfe::ch03::BsPrice(type, kS, kK, kT, kSigma, kR, kQ);
  const double err = std::abs(
      ch08::EuropeanHeatCrankNicolson(type, kS, kK, kT, kSigma, kR, kQ, nx) -
      exact);
  if (!Check(state, err < 0.05, "heat-equation Crank-Nicolson price is off"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch08::EuropeanHeatCrankNicolson(type, kS, kK, kT, kSigma, kR, kQ, nx));
  state.counters["abs_error"] = err;
  state.SetComplexityN(static_cast<std::int64_t>(nx));
}

void BmHeatFd(benchmark::State& state, OptionType type) {
  const auto nx = static_cast<std::size_t>(state.range(0));
  const double exact = p4mfe::ch03::BsPrice(type, kS, kK, kT, kSigma, kR, kQ);
  const double err = std::abs(
      ch08::EuropeanHeatFd(type, kS, kK, kT, kSigma, kR, kQ, nx) - exact);
  if (!Check(state, err < 0.05, "heat-equation FD price is off")) return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch08::EuropeanHeatFd(type, kS, kK, kT, kSigma, kR, kQ, nx));
  state.counters["abs_error"] = err;
  state.SetComplexityN(static_cast<std::int64_t>(nx));
}

void BmBinomial(benchmark::State& state, OptionType type,
                ch08::ExerciseStyle style) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const double v =
      ch08::BinomialPrice(type, kS, kK, kT, kSigma, kR, kQ, n, style);
  const double bs = p4mfe::ch03::BsPrice(type, kS, kK, kT, kSigma, kR, kQ);
  const bool ok = style == ch08::ExerciseStyle::kEuropean
                      ? std::abs(v - bs) < 0.05
                      : v >= bs - 1e-9;
  if (!Check(state, ok, "binomial price is off")) return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch08::BinomialPrice(type, kS, kK, kT, kSigma, kR, kQ, n, style));
  state.counters["price"] = v;
  state.SetComplexityN(static_cast<std::int64_t>(n));
}

void BmEarlyExercisePremium(benchmark::State& state) {
  std::vector<double> strikes(50);
  for (std::size_t i = 0; i < strikes.size(); ++i)
    strikes[i] = 70.0 + static_cast<double>(i);
  for (auto _ : state) {
    double sum = 0.0;
    for (const double k : strikes)
      sum += ch08::EarlyExercisePremium(OptionType::kPut, kS, k, kT, kSigma, kR,
                                        kQ, 500);
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, strikes.size(), "strikes");
}

const bool kRegistered = [] {
  benchmark::RegisterBenchmark("W1/ch08/barriers", BmBarriers);
  benchmark::RegisterBenchmark("W2/ch08/box_muller", BmBoxMullerPairs);

  auto* di =
      benchmark::RegisterBenchmark("W4/ch08/double_integral", BmDoubleIntegral);
  for (const auto n : p4mfe::bench::ByTier<std::vector<std::int64_t>>(
           {50, 200}, {200, 1000}, {1000, 4000}))
    di->Arg(n);
  di->Complexity(benchmark::oN)->Unit(benchmark::kMillisecond);
  auto* pi =
      benchmark::RegisterBenchmark("W4/ch08/polar_integral", BmPolarIntegral);
  for (const auto n : p4mfe::bench::ByTier<std::vector<std::int64_t>>(
           {200}, {200, 1000}, {1000, 4000}))
    pi->Arg(n);
  pi->Unit(benchmark::kMillisecond);

  auto* bm = benchmark::RegisterBenchmark("W5/ch08/box_muller_samples",
                                          BmBoxMullerSamples);
  for (const auto n : p4mfe::bench::ByTier<std::vector<std::int64_t>>(
           {10'000}, {1'000'000, 10'000'000}, {10'000'000, 100'000'000}))
    bm->Arg(n);
  bm->Complexity(benchmark::oN)->Unit(benchmark::kMillisecond);

  for (const auto type : {OptionType::kCall, OptionType::kPut}) {
    const std::string t = type == OptionType::kCall ? "call" : "put";
    auto* h = benchmark::RegisterBenchmark("W6/ch08/european_heat_fd/" + t,
                                           BmHeatFd, type);
    for (const auto n : p4mfe::bench::ByTier<std::vector<std::int64_t>>(
             {100}, {100, 200, 400, 800}, {400, 800, 1600, 3200}))
      h->Arg(n);
    h->Complexity(benchmark::oNCubed)->Unit(benchmark::kMillisecond);
    auto* cn = benchmark::RegisterBenchmark(
        "W6/ch08/european_heat_crank_nicolson/" + t, BmHeatCrankNicolson, type);
    for (const auto n : p4mfe::bench::ByTier<std::vector<std::int64_t>>(
             {100}, {100, 200, 400, 800}, {400, 800, 1600, 3200}))
      cn->Arg(n);
    cn->Complexity(benchmark::oNSquared)->Unit(benchmark::kMillisecond);
  }
  const std::pair<const char*, ch08::ExerciseStyle> styles[] = {
      {"european", ch08::ExerciseStyle::kEuropean},
      {"american", ch08::ExerciseStyle::kAmerican}};
  for (const auto& [sname, style] : styles) {
    auto* b = benchmark::RegisterBenchmark(
        std::string("W6/ch08/binomial_price/put/") + sname, BmBinomial,
        OptionType::kPut, style);
    for (const auto n : p4mfe::bench::ByTier<std::vector<std::int64_t>>(
             {100, 500}, {100, 500, 2'000, 10'000}, {2'000, 10'000, 50'000}))
      b->Arg(n);
    b->Complexity(benchmark::oNSquared)->Unit(benchmark::kMillisecond);
  }
  benchmark::RegisterBenchmark("W6/ch08/early_exercise_premium",
                               BmEarlyExercisePremium)
      ->Unit(benchmark::kMillisecond);
  return true;
}();

}  // namespace
