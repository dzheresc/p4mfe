// Chapter 6 benchmarks:
//   W1  ATM approximations (6.5) over the near-the-money slice of the chain
//   W3  dollar duration/convexity, DV01, portfolio risk, immunization and
//       yield curve shifts over the bond universe
//   W8  returns on a 10-year daily price history for 500 assets
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/bonds.hpp>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch06/bs_atm.hpp>
#include <p4mfe/ch06/dollar_duration.hpp>
#include <p4mfe/ch06/duration_convexity.hpp>
#include <p4mfe/ch06/returns.hpp>
#include <p4mfe/ch06/yield_curve_shifts.hpp>
#include <random>
#include <vector>

namespace {

namespace ch06 = p4mfe::ch06;
using p4mfe::bench::check;
using p4mfe::bench::set_items;

// ----------------------------------------------------------------- W1 -----

void BM_atm(benchmark::State& state) {
  // ATM-forward-ish quotes, zero rates as in the book's approximation.
  std::vector<p4mfe::bench::OptionQuote> atm;
  for (const auto& o :
       p4mfe::bench::make_option_chain(p4mfe::bench::chain_spec_for_tier()))
    if (std::abs(o.log_moneyness) < 0.02) atm.push_back(o);
  if (!check(state, !atm.empty(), "no ATM quotes")) return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : atm) {
      sum += ch06::atm_call_approx(o.S, o.T, o.sigma, 0.045, 0.015);
      sum += ch06::atm_put_approx(o.S, o.T, o.sigma, 0.045, 0.015);
      sum += ch06::atm_implied_vol_approx(
          ch06::atm_price_exact(o.S, o.T, o.sigma), o.S, o.T);
      const auto g = ch06::atm_greeks_approx(o.type, o.S, o.T, o.sigma);
      sum += g.delta + g.gamma + g.vega + g.theta;
      sum += ch06::atm_relative_error(o.T, o.sigma);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, atm.size(), "options");
}

// ----------------------------------------------------------------- W3 -----

const std::vector<p4mfe::bench::Bond>& universe() {
  static const auto u =
      p4mfe::bench::bond_universe(p4mfe::bench::bond_universe_size());
  return u;
}

void BM_dollar_risk(benchmark::State& state) {
  const auto& u = universe();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) {
      const auto r = ch06::bond_dollar_risk(b.cf, 0.05);
      sum += r.dollar_duration + r.dollar_convexity +
             ch06::dv01_central(b.cf, 0.05) +
             ch06::cash_flow_dispersion(b.cf, 0.05) +
             ch06::convexity_discrete(b.cf, 0.05, b.freq);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, u.size(), "bonds");
}

void BM_portfolio_risk(benchmark::State& state) {
  const auto& u = universe();
  std::vector<ch06::DollarRisk> risks;
  std::vector<double> units, values, durations;
  for (std::size_t i = 0; i < u.size(); ++i) {
    risks.push_back(ch06::bond_dollar_risk(u[i].cf, 0.05));
    units.push_back(static_cast<double>(1 + i % 7) * 1000.0);
    values.push_back(risks.back().value * units.back());
    durations.push_back(risks.back().dollar_duration / risks.back().value);
  }
  const auto h1 =
      ch06::bond_dollar_risk(p4mfe::ch02::BondCashFlows(2.0, 0.04, 2), 0.05);
  const auto h2 =
      ch06::bond_dollar_risk(p4mfe::ch02::BondCashFlows(10.0, 0.045, 2), 0.05);
  for (auto _ : state) {
    const auto p = ch06::portfolio_dollar_risk(units, risks);
    auto imm = ch06::immunize(p, h1, h2);
    benchmark::DoNotOptimize(imm);
    benchmark::DoNotOptimize(ch06::portfolio_duration(values, durations));
    benchmark::DoNotOptimize(ch06::value_change_approx_convexity(p, 0.001));
  }
  set_items(state, u.size(), "bonds");
}

void BM_curve_shifts(benchmark::State& state) {
  const auto& u = universe();
  const std::size_t n = std::min<std::size_t>(
      u.size(), p4mfe::bench::by_tier<std::size_t>(100, 1000, 10'000));
  const auto z = [](double t) { return p4mfe::bench::true_zero_rate(t); };
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const auto& cf = u[i].cf;
      sum += ch06::shifted_curve_price(cf, z, 0.001) +
             ch06::shifted_curve_price_approx(cf, z, 0.001);
      sum += ch06::yield_change_from_shift(cf, z, 0.001);
      const auto m = ch06::curve_shift_metrics(cf, z);
      sum += m.duration + m.convexity;
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, n, "bonds");
}

// ----------------------------------------------------------------- W8 -----

struct History {
  std::size_t assets, days;
  std::vector<std::vector<double>> prices;  // per asset
};

const History& history() {
  static const History h = [] {
    History x{p4mfe::bench::by_tier<std::size_t>(10, 500, 2000), 2520, {}};
    std::mt19937_64 rng(42);
    std::normal_distribution<double> z;
    for (std::size_t a = 0; a < x.assets; ++a) {
      std::vector<double> p(x.days);
      p[0] = 100.0;
      for (std::size_t d = 1; d < x.days; ++d)
        p[d] = p[d - 1] * std::exp(0.0003 + 0.015 * z(rng));
      x.prices.push_back(std::move(p));
    }
    return x;
  }();
  return h;
}

void BM_returns(benchmark::State& state) {
  const auto& h = history();
  const std::vector<double> w(h.assets, 1.0 / static_cast<double>(h.assets));
  for (auto _ : state) {
    std::vector<double> R(h.assets), r(h.assets);
    for (std::size_t a = 0; a < h.assets; ++a) {
      const auto pr = ch06::percentage_returns(h.prices[a]);
      const auto lr = ch06::log_returns(h.prices[a]);
      R[a] = ch06::compound_percentage_returns(pr);
      r[a] = ch06::sum_log_returns(lr);
    }
    benchmark::DoNotOptimize(ch06::portfolio_percentage_return(w, R));
    benchmark::DoNotOptimize(ch06::portfolio_log_return(w, r));
    benchmark::DoNotOptimize(ch06::portfolio_log_return_approx(w, r));
  }
  set_items(state, h.assets * h.days, "prices");
}

const bool registered = [] {
  (void)universe();
  (void)history();
  benchmark::RegisterBenchmark("W1/ch06/atm_approximations", BM_atm);
  benchmark::RegisterBenchmark("W3/ch06/dollar_risk", BM_dollar_risk);
  benchmark::RegisterBenchmark("W3/ch06/portfolio_risk+immunize",
                               BM_portfolio_risk);
  benchmark::RegisterBenchmark("W3/ch06/yield_curve_shifts", BM_curve_shifts);
  benchmark::RegisterBenchmark("W8/ch06/returns", BM_returns)
      ->Unit(benchmark::kMillisecond);
  return true;
}();

}  // namespace
