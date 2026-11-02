// Remaining chapter 6 paths (coverage sweep): linearized ATM prices, the
// rest of the dollar-risk toolkit over the bond universe, Greeks-based
// Taylor approximations over the option chain, return conversions, and the
// Taylor polynomial and series helpers.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/bonds.hpp>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch06/bs_atm.hpp>
#include <p4mfe/ch06/dollar_duration.hpp>
#include <p4mfe/ch06/duration_convexity.hpp>
#include <p4mfe/ch06/greeks_taylor.hpp>
#include <p4mfe/ch06/returns.hpp>
#include <p4mfe/ch06/taylor.hpp>
#include <p4mfe/ch06/taylor_series.hpp>
#include <vector>

namespace {

namespace ch03 = p4mfe::ch03;
namespace ch06 = p4mfe::ch06;
using p4mfe::bench::check;
using p4mfe::bench::set_items;

void BmAtmLinearized(benchmark::State& state) {
  const auto chain =
      p4mfe::bench::make_option_chain(p4mfe::bench::chain_spec_for_tier());
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : chain) {
      sum += ch06::AtmCallLinearized(o.S, o.T, o.sigma, 0.045, 0.015);
      sum += ch06::AtmRelativeErrorBound(o.T, o.sigma) +
             ch06::NormCdfLinear(o.log_moneyness);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, chain.size(), "options");
}

void BmDollarRiskRest(benchmark::State& state) {
  const auto u =
      p4mfe::bench::bond_universe(p4mfe::bench::bond_universe_size());
  const auto hedge =
      ch06::BondDollarRisk(p4mfe::ch02::BondCashFlows(10.0, 0.045, 2), 0.05);
  std::vector<double> values(u.size()), convexities(u.size());
  for (std::size_t i = 0; i < u.size(); ++i) {
    values[i] = u[i].price;
    convexities[i] =
        ch06::ConvexityFromDollar(ch06::BondDollarRisk(u[i].cf, 0.05));
  }
  {
    const auto r = ch06::BondDollarRisk(u[0].cf, 0.05);
    if (!check(state,
               std::abs(ch06::Dv01Reprice(u[0].cf, 0.05) - r.Dv01()) <
                   1e-3 * std::abs(r.Dv01()) + 1e-9,
               "DV01 by repricing disagrees with D$/10000"))
      return;
  }
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) {
      const auto r = ch06::BondDollarRisk(b.cf, 0.05);
      sum +=
          ch06::DollarConvexity(b.cf, 0.05) + ch06::ValueChangeApprox(r, 0.001);
      const double duration = ch06::DurationFromDollar(r),
                   convexity = ch06::ConvexityFromDollar(r);
      sum += ch06::DurationDerivative(duration, convexity) +
             ch06::Dv01Reprice(b.cf, 0.05);
      sum += ch06::DurationHedgeUnits(r, hedge);
    }
    sum += ch06::PortfolioConvexity(values, convexities);
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, u.size(), "bonds");
}

void BmGreeksTaylor(benchmark::State& state) {
  const auto spec = p4mfe::bench::chain_spec_for_tier();
  const auto chain = p4mfe::bench::make_option_chain(spec);
  std::vector<ch03::Greeks> g;
  std::vector<double> values;
  for (const auto& o : chain) {
    g.push_back(ch03::BsGreeks(o.type, o.S, o.K, o.T, o.sigma, spec.r, spec.q));
    values.push_back(
        ch03::BsPrice(o.type, o.S, o.K, o.T, o.sigma, spec.r, spec.q));
  }
  {
    const auto& o = chain.front();
    const double th = ch06::ThetaFromPde(values[0], o.S, g[0].delta, g[0].gamma,
                                         o.sigma, spec.r, spec.q);
    if (!check(state,
               std::abs(th - g[0].theta) < 1e-6 * (1.0 + std::abs(g[0].theta)),
               "theta from the PDE is off"))
      return;
  }
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 0; i < chain.size(); ++i) {
      const auto& o = chain[i];
      const double d_spot = 0.01 * o.S;
      sum += ch06::OptionChangeDelta(g[i], d_spot) +
             ch06::OptionChangeDeltaGamma(g[i], d_spot);
      sum += ch06::OptionChangeTaylor(g[i], d_spot, 1.0 / 252.0, 0.01, 0.0005);
      sum += ch06::ThetaFromPde(values[i], o.S, g[i].delta, g[i].gamma, o.sigma,
                                spec.r, spec.q);
      sum += ch06::DeltaNeutralChange(values[i] - g[i].delta * o.S, o.S,
                                      g[i].gamma, o.sigma, spec.r, d_spot,
                                      1.0 / 252.0);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, chain.size(), "options");
}

void BmReturnConversions(benchmark::State& state) {
  // One year of daily returns for 500 assets.
  const std::size_t n = 252 * 500;
  std::vector<double> pct_returns(n), units(500), prices(500);
  for (std::size_t i = 0; i < n; ++i)
    pct_returns[i] = 0.02 * std::sin(0.37 * static_cast<double>(i));
  for (std::size_t i = 0; i < 500; ++i) {
    units[i] = 100.0 + static_cast<double>(i);
    prices[i] = 10.0 + 0.5 * static_cast<double>(i);
  }
  if (!check(state,
             std::abs(ch06::PercentageFromLog(ch06::LogFromPercentage(0.05)) -
                      0.05) < 1e-15,
             "return conversions do not round-trip"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const double x : pct_returns) {
      const double r = ch06::LogFromPercentage(x);
      sum += ch06::PercentageFromLog(r) - ch06::LogReturnTaylor(x, 3);
    }
    const auto w = ch06::PortfolioWeights(units, prices);
    benchmark::DoNotOptimize(sum + w.back());
  }
  set_items(state, n, "returns");
}

void BmTaylor(benchmark::State& state) {
  // Taylor polynomials and remainders for e^x, a 5-asset quadratic in n
  // dimensions, 2-D partials, and the elementary series on a 1,000-point grid.
  const std::vector<double> derivs(12, 1.0);  // e^x at 0
  const auto f = [](const ch06::Vector& v) {
    double s = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i)
      s += std::exp(0.1 * static_cast<double>(i + 1) * v[i]);
    return s;
  };
  const ch06::Vector a(5, 0.0), x(5, 0.1);
  const auto f2 = [](double u, double v) { return std::exp(u) * std::cos(v); };
  if (!check(state,
             std::abs(ch06::TaylorPolynomial(derivs, 0.0, 0.5) -
                      std::exp(0.5)) < 1e-9,
             "taylor_polynomial is off"))
    return;
  const std::size_t n = 1000;
  for (auto _ : state) {
    double sum = ch06::TaylorPolynomial(derivs, 0.0, 0.5);
    const auto ex = [](double t) { return std::exp(t); };
    sum +=
        ch06::TaylorLinear(ex, 0.0, 0.1) + ch06::TaylorQuadratic(ex, 0.0, 0.1);
    sum += ch06::TaylorRemainderBound(std::exp(0.5), 0.0, 0.5, 3);
    sum += ch06::TaylorRemainderIntegral(ex, 0.0, 0.5, 3);
    sum += ch06::TaylorLinearNd(f, a, x) + ch06::TaylorQuadraticNd(f, a, x);
    const auto p = ch06::Partials2d(f2, 0.2, 0.3);
    sum += ch06::TaylorLinear2d(p, 0.01, -0.02) +
           ch06::TaylorQuadratic2d(p, 0.01, -0.02);
    for (std::size_t i = 0; i < n; ++i) {
      const double t =
          -0.9 + 1.8 * static_cast<double>(i) / static_cast<double>(n);
      sum += ch06::ExpSeries(t, 15) + ch06::Log1pSeries(t, 30) +
             ch06::GeometricSeries(t, 30);
      sum += ch06::BinomialSeries(t, 0.5, 20) + ch06::SinSeries(t, 10) +
             ch06::CosSeries(t, 10);
      sum += ch06::NormCdfSeries(t, 15);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, n, "points");
}

const bool kRegistered = [] {
  benchmark::RegisterBenchmark("W1/ch06/atm_linearized", BmAtmLinearized);
  benchmark::RegisterBenchmark("W3/ch06/dollar_risk_rest", BmDollarRiskRest)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W1/ch06/greeks_taylor", BmGreeksTaylor);
  benchmark::RegisterBenchmark("W8/ch06/return_conversions",
                               BmReturnConversions)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W8/ch06/taylor+series", BmTaylor)
      ->Unit(benchmark::kMicrosecond);
  return true;
}();

}  // namespace
