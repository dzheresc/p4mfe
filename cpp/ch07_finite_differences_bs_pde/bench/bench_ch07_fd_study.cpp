// Remaining chapter 7 paths (coverage sweep): the finite-difference accuracy
// study (error tables, observed orders, optimal steps) over 1,000 points, and
// the closed-form PDE solutions and Greeks aggregation over the option chain.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch07/bs_pde.hpp>
#include <p4mfe/ch07/finite_differences.hpp>
#include <vector>

namespace {

namespace ch03 = p4mfe::ch03;
namespace ch07 = p4mfe::ch07;
using p4mfe::bench::check;
using p4mfe::bench::set_items;

void BM_fd_study(benchmark::State& state) {
  const auto f = [](double x) { return std::exp(x) * std::sin(x); };
  const auto df = [](double x) {
    return std::exp(x) * (std::sin(x) + std::cos(x));
  };
  const double p =
      ch07::fd_observed_order(f, 0.5, df(0.5), ch07::FdScheme::Central, 1e-2);
  if (!check(
          state,
          std::abs(p - ch07::truncation_order(ch07::FdScheme::Central)) < 0.1,
          "central order is not 2"))
    return;
  const std::size_t n = 1000;
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double a =
          -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n);
      for (const auto s : {ch07::FdScheme::Forward, ch07::FdScheme::Backward,
                           ch07::FdScheme::Central}) {
        const auto rows = ch07::fd_error_table(f, a, df(a), s, 0.1, 8);
        sum +=
            rows.back().error + ch07::fd_observed_order(f, a, df(a), s, 1e-3);
        sum += ch07::optimal_step(s, 1, a) + ch07::truncation_order(s);
      }
      sum += ch07::backward_difference(
          f, a, ch07::optimal_step(ch07::FdScheme::Backward, 2, a));
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, n, "points");
}

void BM_pde_solutions(benchmark::State& state) {
  const auto spec = p4mfe::bench::chain_spec_for_tier();
  const auto chain = p4mfe::bench::make_option_chain(spec);
  std::vector<ch03::Greeks> gs;
  std::vector<double> w;
  for (const auto& o : chain) {
    gs.push_back(
        ch03::BsGreeks(o.type, o.S, o.K, o.T, o.sigma, spec.r, spec.q));
    w.push_back(o.K > o.S ? 1.0 : -1.0);
  }
  const auto& o0 = chain.front();
  const auto call = [&](double S, double t) {
    return ch03::BsPrice(o0.type, S, o0.K, o0.T - t, o0.sigma, spec.r, spec.q);
  };
  if (!check(state,
             std::abs(ch07::bs_pde_residual_fd(call, o0.S, 0.0, o0.sigma,
                                               spec.r, spec.q)) < 1e-2,
             "Black-Scholes price does not solve the PDE"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : chain) {
      sum +=
          ch07::power_solution(2.0, o.S, 0.0, o.sigma, spec.r, spec.q) * 1e-8;
      sum += ch07::power_solution_rate(-1.0, o.sigma, spec.r, spec.q);
      sum += ch07::forward_contract_value(o.S, 0.0, o.K, o.T, spec.r, spec.q);
    }
    sum += ch07::bs_pde_residual_fd(call, o0.S, 0.0, o0.sigma, spec.r, spec.q);
    const auto g = ch07::combine_greeks(w, gs);
    benchmark::DoNotOptimize(sum + g.delta + g.gamma);
  }
  set_items(state, chain.size(), "options");
}

const bool registered = [] {
  benchmark::RegisterBenchmark("W4/ch07/fd_accuracy_study", BM_fd_study)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W1/ch07/pde_closed_forms+combine_greeks",
                               BM_pde_solutions)
      ->Unit(benchmark::kMicrosecond);
  return true;
}();

}  // namespace
