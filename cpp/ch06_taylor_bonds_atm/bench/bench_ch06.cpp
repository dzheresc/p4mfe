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

void BmAtm(benchmark::State& state) {
  // ATM-forward-ish quotes, zero rates as in the book's approximation.
  std::vector<p4mfe::bench::OptionQuote> atm;
  for (const auto& o :
       p4mfe::bench::make_option_chain(p4mfe::bench::chain_spec_for_tier()))
    if (std::abs(o.log_moneyness) < 0.02) atm.push_back(o);
  if (!check(state, !atm.empty(), "no ATM quotes")) return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : atm) {
      sum += ch06::AtmCallApprox(o.S, o.T, o.sigma, 0.045, 0.015);
      sum += ch06::AtmPutApprox(o.S, o.T, o.sigma, 0.045, 0.015);
      sum += ch06::AtmImpliedVolApprox(ch06::AtmPriceExact(o.S, o.T, o.sigma),
                                       o.S, o.T);
      const auto g = ch06::AtmGreeksApprox(o.type, o.S, o.T, o.sigma);
      sum += g.delta + g.gamma + g.vega + g.theta;
      sum += ch06::AtmRelativeError(o.T, o.sigma);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, atm.size(), "options");
}

// ----------------------------------------------------------------- W3 -----

const std::vector<p4mfe::bench::Bond>& Universe() {
  static const auto kU =
      p4mfe::bench::bond_universe(p4mfe::bench::bond_universe_size());
  return kU;
}

void BmDollarRisk(benchmark::State& state) {
  const auto& u = Universe();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) {
      const auto r = ch06::BondDollarRisk(b.cf, 0.05);
      sum += r.dollar_duration + r.dollar_convexity +
             ch06::Dv01Central(b.cf, 0.05) +
             ch06::CashFlowDispersion(b.cf, 0.05) +
             ch06::ConvexityDiscrete(b.cf, 0.05, b.freq);
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, u.size(), "bonds");
}

void BmPortfolioRisk(benchmark::State& state) {
  const auto& u = Universe();
  std::vector<ch06::DollarRisk> risks;
  std::vector<double> units, values, durations;
  for (std::size_t i = 0; i < u.size(); ++i) {
    risks.push_back(ch06::BondDollarRisk(u[i].cf, 0.05));
    units.push_back(static_cast<double>(1 + i % 7) * 1000.0);
    values.push_back(risks.back().value * units.back());
    durations.push_back(risks.back().dollar_duration / risks.back().value);
  }
  const auto h1 =
      ch06::BondDollarRisk(p4mfe::ch02::BondCashFlows(2.0, 0.04, 2), 0.05);
  const auto h2 =
      ch06::BondDollarRisk(p4mfe::ch02::BondCashFlows(10.0, 0.045, 2), 0.05);
  for (auto _ : state) {
    const auto p = ch06::PortfolioDollarRisk(units, risks);
    auto imm = ch06::Immunize(p, h1, h2);
    benchmark::DoNotOptimize(imm);
    benchmark::DoNotOptimize(ch06::PortfolioDuration(values, durations));
    benchmark::DoNotOptimize(ch06::ValueChangeApproxConvexity(p, 0.001));
  }
  set_items(state, u.size(), "bonds");
}

void BmCurveShifts(benchmark::State& state) {
  const auto& u = Universe();
  const std::size_t n = std::min<std::size_t>(
      u.size(), p4mfe::bench::by_tier<std::size_t>(100, 1000, 10'000));
  const auto z = [](double t) { return p4mfe::bench::true_zero_rate(t); };
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const auto& cf = u[i].cf;
      sum += ch06::ShiftedCurvePrice(cf, z, 0.001) +
             ch06::ShiftedCurvePriceApprox(cf, z, 0.001);
      sum += ch06::YieldChangeFromShift(cf, z, 0.001);
      const auto m = ch06::CurveShiftMetrics(cf, z);
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

const History& GetHistory() {
  static const History kH = [] {
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
  return kH;
}

void BmReturns(benchmark::State& state) {
  const auto& h = GetHistory();
  const std::vector<double> w(h.assets, 1.0 / static_cast<double>(h.assets));
  for (auto _ : state) {
    std::vector<double> pct_returns(h.assets), r(h.assets);
    for (std::size_t a = 0; a < h.assets; ++a) {
      const auto pr = ch06::PercentageReturns(h.prices[a]);
      const auto lr = ch06::LogReturns(h.prices[a]);
      pct_returns[a] = ch06::CompoundPercentageReturns(pr);
      r[a] = ch06::SumLogReturns(lr);
    }
    benchmark::DoNotOptimize(ch06::PortfolioPercentageReturn(w, pct_returns));
    benchmark::DoNotOptimize(ch06::PortfolioLogReturn(w, r));
    benchmark::DoNotOptimize(ch06::PortfolioLogReturnApprox(w, r));
  }
  set_items(state, h.assets * h.days, "prices");
}

const bool kRegistered = [] {
  (void)Universe();
  (void)GetHistory();
  benchmark::RegisterBenchmark("W1/ch06/atm_approximations", BmAtm);
  benchmark::RegisterBenchmark("W3/ch06/dollar_risk", BmDollarRisk);
  benchmark::RegisterBenchmark("W3/ch06/portfolio_risk+immunize",
                               BmPortfolioRisk);
  benchmark::RegisterBenchmark("W3/ch06/yield_curve_shifts", BmCurveShifts);
  benchmark::RegisterBenchmark("W8/ch06/returns", BmReturns)
      ->Unit(benchmark::kMillisecond);
  return true;
}();

}  // namespace
