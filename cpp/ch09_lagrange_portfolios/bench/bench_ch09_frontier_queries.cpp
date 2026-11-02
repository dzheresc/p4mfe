// Remaining chapter 9 paths (coverage sweep): covariance from vols and
// correlations, portfolio statistics, and the frontier queries with and
// without cash, for 50 assets and 100 (smoke) or 500 assets.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <p4mfe/bench/portfolio.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch09/portfolio.hpp>

namespace {

namespace ch09 = p4mfe::ch09;
using p4mfe::bench::check;

void BM_frontier_queries(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto u = p4mfe::bench::asset_universe(n);
  // Vols and correlations back out of Sigma, then rebuild it.
  ch09::Vector sigma(n);
  for (std::size_t i = 0; i < n; ++i) sigma[i] = std::sqrt(u.Sigma(i, i));
  ch09::Matrix corr(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      corr(i, j) = u.Sigma(i, j) / (sigma[i] * sigma[j]);
  const double r_f = 0.03, target = 0.08, sigma_P = 0.15;
  {
    const auto S = ch09::covariance_matrix(sigma, corr);
    const auto w = ch09::min_variance_portfolio(u.mu, u.Sigma, target);
    const double v = ch09::portfolio_std_dev(w, u.Sigma);
    if (!check(state,
               std::abs(S(n - 1, 0) - u.Sigma(n - 1, 0)) < 1e-12 &&
                   std::abs(v * v - ch09::frontier_variance(u.mu, u.Sigma,
                                                            target)) < 1e-9,
               "frontier variance disagrees with the portfolio's"))
      return;
  }
  for (auto _ : state) {
    auto S = ch09::covariance_matrix(sigma, corr);
    auto wc =
        ch09::min_variance_portfolio_with_cash(u.mu, u.Sigma, r_f, target);
    auto wm = ch09::max_return_portfolio(u.mu, u.Sigma, sigma_P);
    auto wmc =
        ch09::max_return_portfolio_with_cash(u.mu, u.Sigma, r_f, sigma_P);
    double sum =
        ch09::portfolio_mean(wm, u.mu) + ch09::portfolio_std_dev(wm, u.Sigma);
    sum += ch09::frontier_variance(u.mu, u.Sigma, target) +
           ch09::sharpe_ratio(wm, u.mu, u.Sigma, r_f);
    sum += ch09::max_return_for_std_dev(u.mu, u.Sigma, sigma_P) +
           ch09::max_return_with_cash(u.mu, u.Sigma, r_f, sigma_P);
    benchmark::DoNotOptimize(S);
    benchmark::DoNotOptimize(wc);
    benchmark::DoNotOptimize(wmc);
    benchmark::DoNotOptimize(sum);
  }
  state.SetComplexityN(static_cast<std::int64_t>(n));
}

const bool registered = [] {
  benchmark::RegisterBenchmark("W7/ch09/frontier_queries", BM_frontier_queries)
      ->Arg(50)
      ->Arg(p4mfe::bench::by_tier<std::int64_t>(100, 500, 500))
      ->Unit(benchmark::kMillisecond);
  return true;
}();

}  // namespace
