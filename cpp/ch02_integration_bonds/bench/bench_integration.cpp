// Workload W4 (docs/performance-testing-plan.md), chapter 2 part: composite
// rules at fixed n (with a complexity fit), the tolerance-driven driver, and
// improper integrals.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch02/integration.hpp>
#include <string>
#include <vector>

namespace {

namespace ch02 = p4mfe::ch02;
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;

// Smooth, peaked and oscillatory integrands on [0, 1].
double Smooth(double x) { return std::exp(-x * x); }
double Peaked(double x) { return 1.0 / (1.0 + 100.0 * (x - 0.5) * (x - 0.5)); }
double Oscillatory(double x) { return std::sin(50.0 * x); }

const double kExactPeaked = 0.2 * std::atan(5.0);
const double kExactOscillatory = (1.0 - std::cos(50.0)) / 50.0;

struct Integrand {
  const char* name;
  double (*f)(double);
};
constexpr Integrand kIntegrands[] = {
    {"smooth", Smooth}, {"peaked", Peaked}, {"oscillatory", Oscillatory}};

void BmRule(benchmark::State& state, double (*f)(double), ch02::Rule rule) {
  const auto n = static_cast<std::size_t>(state.range(0));
  for (auto _ : state)
    benchmark::DoNotOptimize(ch02::Integrate(f, 0.0, 1.0, n, rule));
  state.SetComplexityN(static_cast<std::int64_t>(n));
  SetItems(state, n, "intervals");
}

void BmToTolerance(benchmark::State& state, double (*f)(double), double exact,
                   double tol) {
  const auto res = ch02::IntegrateToTolerance(f, 0.0, 1.0, tol);
  if (!Check(state, std::abs(res.value - exact) < 100.0 * tol,
             "integrate_to_tolerance is off"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(ch02::IntegrateToTolerance(f, 0.0, 1.0, tol));
  state.counters["accepted_n"] = static_cast<double>(res.intervals);
}

void BmGamma(benchmark::State& state, double x) {
  if (!Check(state,
             std::abs(ch02::GammaFunction(x) - std::tgamma(x)) <
                 1e-6 * std::tgamma(x),
             "gamma_function is off"))
    return;
  for (auto _ : state) benchmark::DoNotOptimize(ch02::GammaFunction(x));
}

void BmImproperGaussian(benchmark::State& state) {
  const auto f = [](double x) { return std::exp(-0.5 * x * x); };
  const double exact = std::sqrt(2.0 * std::numbers::pi);
  if (!Check(
          state,
          std::abs(ch02::ImproperIntegral(f, -ch02::kInfinity, ch02::kInfinity)
                       .value -
                   exact) < 1e-7,
          "improper_integral is off"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch02::ImproperIntegral(f, -ch02::kInfinity, ch02::kInfinity));
}

const bool kRegistered = [] {
  const auto ns = p4mfe::bench::ByTier<std::vector<std::int64_t>>(
      {1 << 10, 1 << 12}, {1 << 10, 1 << 14, 1 << 18, 1 << 22},
      {1 << 14, 1 << 18, 1 << 22, 1 << 24});
  const std::pair<const char*, ch02::Rule> rules[] = {
      {"midpoint", ch02::Rule::kMidpoint},
      {"trapezoidal", ch02::Rule::kTrapezoidal},
      {"simpson", ch02::Rule::kSimpson}};
  for (const auto& [rname, rule] : rules)
    for (const auto& g : kIntegrands) {
      auto* b = benchmark::RegisterBenchmark(
          std::string("W4/ch02/") + rname + "/" + g.name, BmRule, g.f, rule);
      for (const auto n : ns) b->Arg(n);
      b->Complexity(benchmark::oN);
    }
  for (const double tol : {1e-6, 1e-9, 1e-12}) {
    const std::string t = tol == 1e-6 ? "1e-6" : tol == 1e-9 ? "1e-9" : "1e-12";
    benchmark::RegisterBenchmark("W4/ch02/integrate_to_tolerance/peaked/" + t,
                                 BmToTolerance, Peaked, kExactPeaked, tol);
    benchmark::RegisterBenchmark(
        "W4/ch02/integrate_to_tolerance/oscillatory/" + t, BmToTolerance,
        Oscillatory, kExactOscillatory, tol);
  }
  for (const double x : {1.5, 5.0, 20.0})
    benchmark::RegisterBenchmark(
        std::string("W4/ch02/gamma_function/") + (x == 1.5   ? "1.5"
                                                  : x == 5.0 ? "5"
                                                             : "20"),
        BmGamma, x);
  benchmark::RegisterBenchmark("W4/ch02/improper_integral/gaussian",
                               BmImproperGaussian);
  return true;
}();

}  // namespace
