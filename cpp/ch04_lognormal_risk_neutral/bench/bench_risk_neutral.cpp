// Workloads W4 and W5 (docs/performance-testing-plan.md), chapter 4 part:
// risk-neutral pricing by quadrature (call, digital, power payoff) and by
// Monte Carlo (antithetic on and off). Monte Carlo memory must stay O(1).
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch04/in_the_money.hpp>
#include <p4mfe/ch04/risk_neutral.hpp>
#include <string>
#include <vector>

namespace {

namespace ch04 = p4mfe::ch04;
using p4mfe::bench::check;
using p4mfe::bench::set_items;
using p4mfe::ch01::OptionType;

constexpr double kS0 = 5000.0, kK = 5100.0, kT = 0.5, kSigma = 0.2, kR = 0.045,
                 kQ = 0.015;

void BmQuadratureCall(benchmark::State& state) {
  const auto call = [](double s) { return std::max(s - kK, 0.0); };
  const double exact =
      p4mfe::ch03::BsPrice(OptionType::kCall, kS0, kK, kT, kSigma, kR, kQ);
  if (!check(state,
             std::abs(ch04::RiskNeutralPrice(call, kS0, kT, kSigma, kR, kQ) -
                      exact) < 1e-5 * kS0,
             "quadrature call price is off"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch04::RiskNeutralPrice(call, kS0, kT, kSigma, kR, kQ));
}

void BmQuadratureDigital(benchmark::State& state) {
  const auto digital = [](double s) { return s > kK ? 1.0 : 0.0; };
  const double exact =
      ch04::CashOrNothing(OptionType::kCall, kS0, kK, kT, kSigma, kR, kQ);
  if (!check(state,
             std::abs(ch04::RiskNeutralPrice(digital, kS0, kT, kSigma, kR, kQ) -
                      exact) < 1e-4,
             "quadrature digital price is off"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch04::RiskNeutralPrice(digital, kS0, kT, kSigma, kR, kQ));
}

void BmQuadraturePower(benchmark::State& state) {
  // E[(S_T / S0)^2] = exp((2(r - q) + sigma^2) T). Scaled by S0 so the
  // absolute tolerance of the quadrature is meaningful.
  const auto sq = [](double s) { return (s / kS0) * (s / kS0); };
  const double exact = std::exp((2.0 * (kR - kQ) + kSigma * kSigma) * kT);
  if (!check(
          state,
          std::abs(ch04::RiskNeutralExpectation(sq, kS0, kT, kSigma, kR, kQ) -
                   exact) < 1e-6 * exact,
          "quadrature power payoff is off"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch04::RiskNeutralExpectation(sq, kS0, kT, kSigma, kR, kQ));
}

void BmClosedForms(benchmark::State& state) {
  // The chapter's closed forms over 1000 strikes: rn_expected_payoff,
  // digitals and exercise probabilities.
  std::vector<double> strikes(1000);
  for (std::size_t i = 0; i < strikes.size(); ++i)
    strikes[i] = kS0 * (0.5 + static_cast<double>(i) / 1000.0);
  for (auto _ : state) {
    double sum = 0.0;
    for (const double k : strikes) {
      sum +=
          ch04::RnExpectedPayoff(OptionType::kCall, kS0, k, kT, kSigma, kR, kQ);
      sum += ch04::CashOrNothing(OptionType::kPut, kS0, k, kT, kSigma, kR, kQ);
      sum +=
          ch04::AssetOrNothing(OptionType::kCall, kS0, k, kT, kSigma, kR, kQ);
      auto p = ch04::ComputeExerciseProbabilities(kS0, k, kT, kSigma, kR, kQ);
      benchmark::DoNotOptimize(p);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, strikes.size(), "strikes");
}

void BmMonteCarlo(benchmark::State& state, bool antithetic) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto call = [](double s) { return std::max(s - kK, 0.0); };
  const double exact =
      p4mfe::ch03::BsPrice(OptionType::kCall, kS0, kK, kT, kSigma, kR, kQ);
  const auto res =
      ch04::MonteCarloPrice(call, kS0, kT, kSigma, kR, kQ, n, 42, antithetic);
  if (!check(state, std::abs(res.price - exact) < 5.0 * res.std_error + 1e-9,
             "Monte Carlo price off by > 5 SE"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(ch04::MonteCarloPrice(call, kS0, kT, kSigma, kR,
                                                   kQ, n, 42, antithetic));
  state.counters["std_error"] = res.std_error;
  state.SetComplexityN(static_cast<std::int64_t>(n));
  set_items(state, n, "paths");
}

const bool kRegistered = [] {
  benchmark::RegisterBenchmark("W4/ch04/risk_neutral_price/call",
                               BmQuadratureCall);
  benchmark::RegisterBenchmark("W4/ch04/risk_neutral_price/digital",
                               BmQuadratureDigital);
  benchmark::RegisterBenchmark("W4/ch04/risk_neutral_expectation/power",
                               BmQuadraturePower);
  benchmark::RegisterBenchmark("W1/ch04/closed_forms", BmClosedForms);
  const auto ns = p4mfe::bench::by_tier<std::vector<std::int64_t>>(
      {10'000}, {10'000, 1'000'000, 10'000'000},
      {1'000'000, 10'000'000, 100'000'000});
  for (const bool anti : {true, false}) {
    auto* b =
        benchmark::RegisterBenchmark(std::string("W5/ch04/monte_carlo_price/") +
                                         (anti ? "antithetic" : "plain"),
                                     BmMonteCarlo, anti);
    for (const auto n : ns) b->Arg(n);
    b->Complexity(benchmark::oN)->Unit(benchmark::kMillisecond);
  }
  return true;
}();

}  // namespace
