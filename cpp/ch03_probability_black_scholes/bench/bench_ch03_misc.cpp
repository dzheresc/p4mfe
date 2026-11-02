// Remaining chapter 3 paths (coverage sweep): sample statistics on long
// return series, distributions by quadrature, normal and lognormal closed
// forms, hedging arithmetic and the Black-Scholes identity checks over the
// option chain.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/hedging.hpp>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/ch03/probability.hpp>
#include <random>
#include <vector>

namespace {

namespace ch03 = p4mfe::ch03;
using p4mfe::bench::check;
using p4mfe::bench::set_items;

void BM_sample_statistics(benchmark::State& state) {
  // Two correlated daily return series (rho = 0.6).
  const std::size_t n =
      p4mfe::bench::by_tier<std::size_t>(10'000, 1'000'000, 10'000'000);
  std::mt19937_64 rng(42);
  std::normal_distribution<double> z;
  std::vector<double> x(n), y(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double a = z(rng), b = z(rng);
    x[i] = 0.01 * a;
    y[i] = 0.01 * (0.6 * a + 0.8 * b);
  }
  if (!check(state, std::abs(ch03::sample_correlation(x, y) - 0.6) < 0.05,
             "sample correlation is off"))
    return;
  for (auto _ : state) {
    benchmark::DoNotOptimize(ch03::sample_mean(x));
    benchmark::DoNotOptimize(ch03::sample_variance(x));
    benchmark::DoNotOptimize(ch03::sample_covariance(x, y));
    benchmark::DoNotOptimize(ch03::sample_correlation(x, y));
  }
  set_items(state, n, "observations");
}

void BM_distributions(benchmark::State& state) {
  // Discrete: a 1,000-point distribution; continuous: exponential by
  // quadrature.
  std::vector<double> v(1000), p(1000, 1.0 / 1000.0);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<double>(i);
  const ch03::DiscreteDistribution d(v, p);
  const auto e = ch03::exponential_distribution(2.0);
  std::vector<ch03::JointOutcome> joint;
  for (int i = 0; i < 100; ++i)
    joint.push_back(
        {static_cast<double>(i), static_cast<double>(i * i % 17), 0.01});
  if (!check(state, std::abs(ch03::mean(e) - 0.5) < 1e-6,
             "exponential mean is off"))
    return;
  for (auto _ : state) {
    benchmark::DoNotOptimize(ch03::mean(d) + ch03::variance(d));
    benchmark::DoNotOptimize(ch03::mean(e) + ch03::variance(e) +
                             ch03::cdf(e, 1.0));
    benchmark::DoNotOptimize(ch03::covariance(joint) +
                             ch03::correlation(joint));
    benchmark::DoNotOptimize(ch03::variance_of_sum(1.0, 0.04, 2.0, 0.09, 0.01));
  }
}

void BM_normal_closed_forms(benchmark::State& state) {
  const std::size_t n =
      p4mfe::bench::by_tier<std::size_t>(1'000, 100'000, 1'000'000);
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double x =
          50.0 + 100.0 * static_cast<double>(i) / static_cast<double>(n);
      sum +=
          ch03::normal_pdf(x, 100.0, 20.0) + ch03::normal_cdf(x, 100.0, 20.0);
      sum +=
          ch03::lognormal_pdf(x, 4.6, 0.2) + ch03::lognormal_cdf(x, 4.6, 0.2);
      sum += ch03::lognormal_mean(4.6, 0.2) +
             ch03::lognormal_variance(4.6, 0.2) + ch03::std_normal_mgf(0.3);
      const auto lp = ch03::log_price_distribution(100.0, 0.5, 0.2, 0.08, 0.01);
      sum += lp.mean + ch03::prob_price_above(x, 100.0, 0.5, 0.2, 0.08, 0.01);
      sum += ch03::linear_combination(1.0, {0.0, 1.0}, 2.0, {1.0, 4.0}, 0.3)
                 .variance;
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, n, "points");
}

void BM_hedging_and_identities(benchmark::State& state) {
  const auto chain =
      p4mfe::bench::make_option_chain(p4mfe::bench::chain_spec_for_tier());
  const p4mfe::bench::ChainSpec spec;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : chain) {
      const auto g =
          ch03::bs_greeks(o.type, o.S, o.K, o.T, o.sigma, spec.r, spec.q);
      const double V =
          ch03::bs_price(o.type, o.S, o.K, o.T, o.sigma, spec.r, spec.q);
      const auto h = ch03::delta_hedge(-1000.0, V, g.delta, o.S);
      const auto dg = ch03::delta_gamma_hedge(-1000.0 * g.delta,
                                              -1000.0 * g.gamma, 0.5, 0.001);
      sum += h.cash + dg.shares + ch03::delta_hedge_shares(g.delta);
      sum += ch03::portfolio_change_approx(g.delta, g.gamma, 1.0, g.theta,
                                           1.0 / 252.0);
      sum +=
          ch03::bs_pde_residual(o.type, o.S, o.K, o.T, o.sigma, spec.r, spec.q);
      sum += ch03::bs_phi_identity_gap(o.S, o.K, o.T, o.sigma, spec.r, spec.q);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, chain.size(), "options");
}

const bool registered = [] {
  benchmark::RegisterBenchmark("W8/ch03/sample_statistics",
                               BM_sample_statistics)
      ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("W4/ch03/distributions", BM_distributions)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W2/ch03/normal_lognormal_closed_forms",
                               BM_normal_closed_forms);
  benchmark::RegisterBenchmark("W1/ch03/hedging+identities",
                               BM_hedging_and_identities);
  return true;
}();

}  // namespace
