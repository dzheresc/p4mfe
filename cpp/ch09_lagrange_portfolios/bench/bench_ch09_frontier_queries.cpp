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

void BmFrontierQueries(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto u = p4mfe::bench::asset_universe(n);
  // Vols and correlations back out of Sigma, then rebuild it.
  ch09::Vector sigma(n);
  for (std::size_t i = 0; i < n; ++i) sigma[i] = std::sqrt(u.Sigma(i, i));
  ch09::Matrix corr(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      corr(i, j) = u.Sigma(i, j) / (sigma[i] * sigma[j]);
  const double r_f = 0.03, target = 0.08, sigma_p = 0.15;
  {
    const auto covariance = ch09::CovarianceMatrix(sigma, corr);
    const auto w = ch09::MinVariancePortfolio(u.mu, u.Sigma, target);
    const double v = ch09::PortfolioStdDev(w, u.Sigma);
    if (!check(state,
               std::abs(covariance(n - 1, 0) - u.Sigma(n - 1, 0)) < 1e-12 &&
                   std::abs(v * v - ch09::FrontierVariance(u.mu, u.Sigma,
                                                           target)) < 1e-9,
               "frontier variance disagrees with the portfolio's"))
      return;
  }
  for (auto _ : state) {
    auto covariance = ch09::CovarianceMatrix(sigma, corr);
    auto wc = ch09::MinVariancePortfolioWithCash(u.mu, u.Sigma, r_f, target);
    auto wm = ch09::MaxReturnPortfolio(u.mu, u.Sigma, sigma_p);
    auto wmc = ch09::MaxReturnPortfolioWithCash(u.mu, u.Sigma, r_f, sigma_p);
    double sum =
        ch09::PortfolioMean(wm, u.mu) + ch09::PortfolioStdDev(wm, u.Sigma);
    sum += ch09::FrontierVariance(u.mu, u.Sigma, target) +
           ch09::SharpeRatio(wm, u.mu, u.Sigma, r_f);
    sum += ch09::MaxReturnForStdDev(u.mu, u.Sigma, sigma_p) +
           ch09::MaxReturnWithCash(u.mu, u.Sigma, r_f, sigma_p);
    benchmark::DoNotOptimize(covariance);
    benchmark::DoNotOptimize(wc);
    benchmark::DoNotOptimize(wmc);
    benchmark::DoNotOptimize(sum);
  }
  state.SetComplexityN(static_cast<std::int64_t>(n));
}

const bool kRegistered = [] {
  benchmark::RegisterBenchmark("W7/ch09/frontier_queries", BmFrontierQueries)
      ->Arg(50)
      ->Arg(p4mfe::bench::by_tier<std::int64_t>(100, 500, 500))
      ->Unit(benchmark::kMillisecond);
  return true;
}();

}  // namespace
