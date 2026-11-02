// Chapter 7 benchmarks:
//   W1  bump-and-reprice Greeks (central and forward) over the chain
//   W5  delta-hedging simulation (daily and hourly rebalancing)
//   W6  ODE solvers (forward/backward Euler, leapfrog) and the tridiagonal
//       boundary value solver, with complexity fits; output size is O(n)
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch07/bs_pde.hpp>
#include <p4mfe/ch07/fd_greeks.hpp>
#include <p4mfe/ch07/ode.hpp>
#include <string>
#include <vector>

namespace {

namespace ch07 = p4mfe::ch07;
using p4mfe::bench::check;
using p4mfe::bench::set_items;
using p4mfe::ch01::OptionType;

// ----------------------------------------------------------------- W1 -----

const std::vector<p4mfe::bench::OptionQuote>& chain() {
  static const auto c =
      p4mfe::bench::make_option_chain(p4mfe::bench::chain_spec_for_tier());
  return c;
}

void BM_fd_greeks(benchmark::State& state, ch07::FdScheme scheme) {
  const auto& c = chain();
  const p4mfe::bench::ChainSpec spec;
  // Delta from bumping should match the closed form on every quote.
  double worst = 0.0;
  for (const auto& o : c) {
    const auto fd = ch07::bs_fd_greeks(o.type, o.S, o.K, o.T, o.sigma, spec.r,
                                       spec.q, {}, scheme);
    const auto cf =
        p4mfe::ch03::bs_greeks(o.type, o.S, o.K, o.T, o.sigma, spec.r, spec.q);
    worst = std::max(worst, std::abs(fd.delta - cf.delta));
  }
  if (!check(state, worst < 1e-4,
             "finite-difference delta disagrees with the closed form"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : c) {
      const auto g = ch07::bs_fd_greeks(o.type, o.S, o.K, o.T, o.sigma, spec.r,
                                        spec.q, {}, scheme);
      sum += g.delta + g.gamma + g.vega + g.theta + g.rho;
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, c.size(), "options");
  state.counters["max_delta_error"] = worst;
}

void BM_pde_residual(benchmark::State& state) {
  const auto& c = chain();
  const p4mfe::bench::ChainSpec spec;
  for (auto _ : state) {
    double worst = 0.0;
    for (const auto& o : c) {
      const auto g = p4mfe::ch03::bs_greeks(o.type, o.S, o.K, o.T, o.sigma,
                                            spec.r, spec.q);
      const double V =
          p4mfe::ch03::bs_price(o.type, o.S, o.K, o.T, o.sigma, spec.r, spec.q);
      worst = std::max(worst, std::abs(ch07::pde_residual(V, g, o.S, o.sigma,
                                                          spec.r, spec.q)));
    }
    benchmark::DoNotOptimize(worst);
  }
  set_items(state, c.size(), "options");
}

// ----------------------------------------------------------------- W5 -----

void BM_delta_hedge(benchmark::State& state) {
  const auto steps = static_cast<std::size_t>(state.range(0));
  const auto paths = static_cast<std::size_t>(state.range(1));
  const auto stats =
      ch07::delta_hedge_stats(OptionType::kCall, 100.0, 100.0, 1.0, 0.2, 0.045,
                              0.015, 0.10, steps, paths);
  // Hedging error ~ O(1/sqrt(steps)) of the premium; the mean is ~0.
  if (!check(state,
             std::abs(stats.mean) <
                 5.0 * stats.std_dev / std::sqrt(static_cast<double>(paths)) +
                     0.05,
             "delta hedge P&L mean is not ~0"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch07::delta_hedge_stats(OptionType::kCall, 100.0, 100.0, 1.0, 0.2,
                                0.045, 0.015, 0.10, steps, paths));
  state.counters["pnl_std_dev"] = stats.std_dev;
  set_items(state, paths, "paths");
}

// ----------------------------------------------------------------- W6 -----

// y' = -2 x y, y(0) = 1, on [0, 2]: y = exp(-x^2).
double ode_rhs(double x, double y) { return -2.0 * x * y; }

void BM_ode(benchmark::State& state, ch07::OdeScheme scheme) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto s = ch07::solve_ivp(ode_rhs, 0.0, 1.0, 2.0, n, scheme);
  const double err =
      ch07::max_error(s, [](double x) { return std::exp(-x * x); });
  if (!check(state, err < 1e-1, "ODE solution is off")) return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch07::solve_ivp(ode_rhs, 0.0, 1.0, 2.0, n, scheme));
  state.counters["max_error"] = err;
  state.SetComplexityN(static_cast<std::int64_t>(n));
  set_items(state, n, "steps");
}

// y'' + y = 0 on [0, pi/2], y(0) = 0, y(pi/2) = 1: y = sin x.
void BM_bvp(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto zero = [](double) { return 0.0; };
  const auto one = [](double) { return 1.0; };
  const double b = std::numbers::pi / 2.0;
  const auto s = ch07::solve_bvp(zero, one, zero, 0.0, b, 0.0, 1.0, n);
  const double err = ch07::max_error(s, [](double x) { return std::sin(x); });
  if (!check(state, err < 1e-3, "BVP solution is off")) return;
  for (auto _ : state)
    benchmark::DoNotOptimize(
        ch07::solve_bvp(zero, one, zero, 0.0, b, 0.0, 1.0, n));
  state.counters["max_error"] = err;
  state.SetComplexityN(static_cast<std::int64_t>(n));
  set_items(state, n, "intervals");
}

const bool registered = [] {
  (void)chain();
  benchmark::RegisterBenchmark("W1/ch07/bs_fd_greeks/central", BM_fd_greeks,
                               ch07::FdScheme::Central);
  benchmark::RegisterBenchmark("W1/ch07/bs_fd_greeks/forward", BM_fd_greeks,
                               ch07::FdScheme::Forward);
  benchmark::RegisterBenchmark("W1/ch07/pde_residual", BM_pde_residual);

  auto* h =
      benchmark::RegisterBenchmark("W5/ch07/delta_hedge_stats", BM_delta_hedge);
  h->ArgNames({"steps", "paths"});
  for (const auto& [steps, paths] : p4mfe::bench::by_tier<
           std::vector<std::pair<std::int64_t, std::int64_t>>>(
           {{52, 100}}, {{252, 10'000}, {252 * 7, 1'000}},
           {{252, 100'000}, {252 * 7, 10'000}}))
    h->Args({steps, paths});
  h->Unit(benchmark::kMillisecond);

  const auto ode_ns = p4mfe::bench::by_tier<std::vector<std::int64_t>>(
      {1'000}, {1'000, 100'000, 1'000'000}, {100'000, 1'000'000, 10'000'000});
  const std::pair<const char*, ch07::OdeScheme> schemes[] = {
      {"euler_forward", ch07::OdeScheme::Forward},
      {"euler_backward", ch07::OdeScheme::Backward},
      {"leapfrog", ch07::OdeScheme::Central}};
  for (const auto& [name, scheme] : schemes) {
    auto* b = benchmark::RegisterBenchmark(std::string("W6/ch07/") + name,
                                           BM_ode, scheme);
    for (const auto n : ode_ns) b->Arg(n);
    b->Complexity(benchmark::oN)->Unit(benchmark::kMicrosecond);
  }
  auto* bvp = benchmark::RegisterBenchmark("W6/ch07/solve_bvp", BM_bvp);
  for (const auto n : p4mfe::bench::by_tier<std::vector<std::int64_t>>(
           {1'000}, {1'000, 100'000, 1'000'000}, {1'000'000, 10'000'000}))
    bvp->Arg(n);
  bvp->Complexity(benchmark::oN)->Unit(benchmark::kMicrosecond);
  return true;
}();

}  // namespace
