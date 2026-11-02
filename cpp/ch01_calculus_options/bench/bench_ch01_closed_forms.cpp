// Remaining chapter 1 paths (coverage sweep, docs/performance-testing-plan.md
// "Covering all code paths"): compounding, forwards, put-call parity
// conversions and arbitrage detection over the option chain, l'Hopital limits
// and differentiation under the integral sign.
#include <benchmark/benchmark.h>

#include <cmath>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch01/forwards.hpp>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <vector>

namespace {

namespace ch01 = p4mfe::ch01;
using p4mfe::bench::set_items;

void BM_parity_sweep(benchmark::State& state) {
  const auto chain =
      p4mfe::bench::make_option_chain(p4mfe::bench::chain_spec_for_tier());
  const p4mfe::bench::ChainSpec spec;
  std::vector<double> calls, puts;
  for (const auto& o : chain) {
    calls.push_back(
        p4mfe::ch03::bs_call(o.S, o.K, o.T, o.sigma, spec.r, spec.q));
    puts.push_back(p4mfe::ch03::bs_put(o.S, o.K, o.T, o.sigma, spec.r, spec.q) +
                   0.01);  // a small mispricing
  }
  for (auto _ : state) {
    double sum = 0.0;
    std::size_t arbs = 0;
    for (std::size_t i = 0; i < chain.size(); ++i) {
      const auto& o = chain[i];
      sum += ch01::put_from_call(calls[i], o.S, o.K, o.T, spec.r, spec.q);
      sum += ch01::call_from_put(puts[i], o.S, o.K, o.T, spec.r, spec.q);
      sum += ch01::synthetic_forward_value(calls[i], puts[i]) +
             ch01::long_forward_value(o.S, o.K, o.T, spec.r, spec.q);
      sum += ch01::compounding_factor(spec.r, o.T, 12.0) +
             ch01::continuous_compounding_factor(spec.r, o.T);
      if (ch01::parity_arbitrage(calls[i], puts[i], o.S, o.K, o.T, spec.r,
                                 spec.q, 1e-6))
        ++arbs;
    }
    benchmark::DoNotOptimize(sum);
    benchmark::DoNotOptimize(arbs);
  }
  set_items(state, chain.size(), "options");
}

void BM_calculus_misc(benchmark::State& state) {
  // d/dt integral_0^t e^(-x t) dx and lim_{x->0} sin x / x.
  const auto f = [](double x, double t) { return std::exp(-x * t); };
  const auto a = [](double) { return 0.0; };
  const auto b = [](double t) { return t; };
  for (auto _ : state) {
    benchmark::DoNotOptimize(ch01::integral_derivative(f, a, b, 1.5));
    benchmark::DoNotOptimize(ch01::lhopital_limit(
        [](double x) { return std::sin(x); }, [](double x) { return x; }, 0.0));
  }
}

const bool registered = [] {
  benchmark::RegisterBenchmark("W1/ch01/parity_and_forwards", BM_parity_sweep);
  benchmark::RegisterBenchmark("W8/ch01/integral_derivative+lhopital_limit",
                               BM_calculus_misc)
      ->Unit(benchmark::kMicrosecond);
  return true;
}();

}  // namespace
