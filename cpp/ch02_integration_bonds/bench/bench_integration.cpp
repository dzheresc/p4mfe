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
using p4mfe::bench::check;
using p4mfe::bench::set_items;

// Smooth, peaked and oscillatory integrands on [0, 1].
double smooth(double x) { return std::exp(-x * x); }
double peaked(double x) { return 1.0 / (1.0 + 100.0 * (x - 0.5) * (x - 0.5)); }
double oscillatory(double x) { return std::sin(50.0 * x); }

const double exact_peaked = 0.2 * std::atan(5.0);
const double exact_oscillatory = (1.0 - std::cos(50.0)) / 50.0;

struct Integrand {
  const char* name;
  double (*f)(double);
};
constexpr Integrand integrands[] = {
    {"smooth", smooth}, {"peaked", peaked}, {"oscillatory", oscillatory}};

void BM_rule(benchmark::State& state, double (*f)(double), ch02::Rule rule) {
  const auto n = static_cast<std::size_t>(state.range(0));
  for (auto _ : state)
    benchmark::DoNotOptimize(ch02::integrate(f, 0.0, 1.0, n, rule));
  state.SetComplexityN(static_cast<std::int64_t>(n));
  set_items(state, n, "intervals");
}

void BM_to_tolerance(benchmark::State& state, double (*f)(double), double exact,
                     double tol) {
  const auto res = ch02::integrate_to_tolerance(f, 0.0, 1.0, tol);
  if (!check(state, std::abs(res.value - exact) < 100.0 * tol,
             "integrate_to_tolerance is off"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(ch02::integrate_to_tolerance(f, 0.0, 1.0, tol));
  state.counters["accepted_n"] = static_cast<double>(res.intervals);
}

void BM_gamma(benchmark::State& state, double x) {
  if (!check(state,
             std::abs(ch02::gamma_function(x) - std::tgamma(x)) <
                 1e-6 * std::tgamma(x),
             "gamma_function is off"))
    return;
  for (auto _ : state) benchmark::DoNotOptimize(ch02::gamma_function(x));
}

void BM_improper_gaussian(benchmark::State& state) {
  const auto f = [](double x) { return std::exp(-0.5 * x * x); };
  const double exact = std::sqrt(2.0 * std::numbers::pi);
  if (!check(
          state,
          std::abs(ch02::improper_integral(f, -ch02::infinity, ch02::infinity)
                       .value -
                   exact) < 1e-7,
          "improper_integral is off"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch02::improper_integral(f, -ch02::infinity, ch02::infinity));
}

const bool registered = [] {
  const auto ns = p4mfe::bench::by_tier<std::vector<std::int64_t>>(
      {1 << 10, 1 << 12}, {1 << 10, 1 << 14, 1 << 18, 1 << 22},
      {1 << 14, 1 << 18, 1 << 22, 1 << 24});
  const std::pair<const char*, ch02::Rule> rules[] = {
      {"midpoint", ch02::Rule::Midpoint},
      {"trapezoidal", ch02::Rule::Trapezoidal},
      {"simpson", ch02::Rule::Simpson}};
  for (const auto& [rname, rule] : rules)
    for (const auto& g : integrands) {
      auto* b = benchmark::RegisterBenchmark(
          std::string("W4/ch02/") + rname + "/" + g.name, BM_rule, g.f, rule);
      for (const auto n : ns) b->Arg(n);
      b->Complexity(benchmark::oN);
    }
  for (const double tol : {1e-6, 1e-9, 1e-12}) {
    const std::string t = tol == 1e-6 ? "1e-6" : tol == 1e-9 ? "1e-9" : "1e-12";
    benchmark::RegisterBenchmark("W4/ch02/integrate_to_tolerance/peaked/" + t,
                                 BM_to_tolerance, peaked, exact_peaked, tol);
    benchmark::RegisterBenchmark(
        "W4/ch02/integrate_to_tolerance/oscillatory/" + t, BM_to_tolerance,
        oscillatory, exact_oscillatory, tol);
  }
  for (const double x : {1.5, 5.0, 20.0})
    benchmark::RegisterBenchmark(
        std::string("W4/ch02/gamma_function/") + (x == 1.5   ? "1.5"
                                                  : x == 5.0 ? "5"
                                                             : "20"),
        BM_gamma, x);
  benchmark::RegisterBenchmark("W4/ch02/improper_integral/gaussian",
                               BM_improper_gaussian);
  return true;
}();

}  // namespace
