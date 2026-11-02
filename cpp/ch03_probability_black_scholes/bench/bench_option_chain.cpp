// Workload W1 (docs/performance-testing-plan.md): Black-Scholes pricing,
// Greeks and implied volatility over an index option chain.
//
// Every benchmark sweeps the whole chain (or one slice of it) per iteration
// and reports options/second. Implied vol benchmarks also report the mean
// iteration count. Setup checks once that implied vol recovers the vol each
// price was generated with, so a fast wrong answer fails loudly.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/implied_vol.hpp>
#include <string>
#include <vector>

namespace {

using p4mfe::bench::OptionQuote;
using p4mfe::ch01::OptionType;

// Slices of the chain that take different paths through the solvers.
enum class Slice { kAll, kAtm, kWings, kShort };

const char* SliceName(Slice s) {
  switch (s) {
    case Slice::kAll:
      return "all";
    case Slice::kAtm:
      return "atm";
    case Slice::kWings:
      return "wings";
    case Slice::kShort:
      return "short";
  }
  return "?";
}

bool InSlice(const OptionQuote& o, Slice s) {
  switch (s) {
    case Slice::kAll:
      return true;
    case Slice::kAtm:
      return std::abs(o.log_moneyness) < 0.05;
    case Slice::kWings:
      return std::abs(o.log_moneyness) > 0.3;
    case Slice::kShort:
      return o.T <= 1.0 / 12.0;
  }
  return false;
}

// All data is built before any benchmark runs (forced from the registration
// below), so the allocation counters measure the kernels, not the setup.
struct CallPut {
  const OptionQuote* call;
  double call_price, put_price;
};

struct Market {
  p4mfe::bench::ChainSpec spec;
  std::vector<OptionQuote> chain;
  std::vector<OptionQuote> slices[4];
  std::vector<CallPut>
      pairs;  // call/put with the same underlying, strike and expiry
};

const Market& GetMarket() {
  static const Market kM = [] {
    Market x;
    x.spec = p4mfe::bench::chain_spec_for_tier();
    x.chain = p4mfe::bench::make_option_chain(x.spec);
    for (const Slice s :
         {Slice::kAll, Slice::kAtm, Slice::kWings, Slice::kShort})
      for (const auto& o : x.chain)
        if (InSlice(o, s)) x.slices[static_cast<int>(s)].push_back(o);
    for (std::size_t i = 0; i + 1 < x.chain.size(); ++i) {
      const auto& c = x.chain[i];
      const auto& p = x.chain[i + 1];
      if (c.type == OptionType::kCall && p.type == OptionType::kPut &&
          c.K == p.K && c.T == p.T && c.S == p.S)
        x.pairs.push_back({&c, c.price, p.price});
    }
    return x;
  }();
  return kM;
}

void SetItemsOptions(benchmark::State& state, std::size_t per_iteration) {
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                          static_cast<std::int64_t>(per_iteration));
  state.counters["options"] = static_cast<double>(per_iteration);
}

// ------------------------------------------------------------ pricing ------

void BmBsPrice(benchmark::State& state) {
  const auto& m = GetMarket();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : m.chain)
      sum += p4mfe::ch03::BsPrice(o.type, o.S, o.K, o.T, o.sigma, m.spec.r,
                                  m.spec.q);
    benchmark::DoNotOptimize(sum);
  }
  SetItemsOptions(state, m.chain.size());
}

void BmBsGreeks(benchmark::State& state) {
  const auto& m = GetMarket();
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : m.chain) {
      const auto g = p4mfe::ch03::BsGreeks(o.type, o.S, o.K, o.T, o.sigma,
                                           m.spec.r, m.spec.q);
      sum += g.delta + g.gamma + g.vega + g.theta + g.rho;
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItemsOptions(state, m.chain.size());
}

// Put-call parity residual (ch01) for every call/put pair in the chain.
void BmParityViolation(benchmark::State& state) {
  const auto& m = GetMarket();
  for (const auto& cp : m.pairs) {
    const auto& o = *cp.call;
    const double v = p4mfe::ch01::ParityViolation(
        cp.call_price, cp.put_price, o.S, o.K, o.T, m.spec.r, m.spec.q);
    if (std::abs(v) > 1e-8 * o.S) {
      state.SkipWithError("put-call parity does not hold on model prices");
      return;
    }
  }
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& cp : m.pairs) {
      const auto& o = *cp.call;
      sum += p4mfe::ch01::ParityViolation(cp.call_price, cp.put_price, o.S, o.K,
                                          o.T, m.spec.r, m.spec.q);
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItemsOptions(state, m.pairs.size());
}

// --------------------------------------------------------- implied vol -----

using IvSolver = p4mfe::ch03::ImpliedVolResult (*)(OptionType, double, double,
                                                   double, double, double,
                                                   double);

p4mfe::ch03::ImpliedVolResult IvNewton(OptionType t, double p, double spot,
                                       double strike, double maturity, double r,
                                       double q) {
  return p4mfe::ch03::ImpliedVolNewton(t, p, spot, strike, maturity, r, q);
}
p4mfe::ch03::ImpliedVolResult IvBisection(OptionType t, double p, double spot,
                                          double strike, double maturity,
                                          double r, double q) {
  return p4mfe::ch03::ImpliedVolBisection(t, p, spot, strike, maturity, r, q);
}

void BmImpliedVol(benchmark::State& state, IvSolver solve, Slice s) {
  const auto& m = GetMarket();
  const auto& quotes = m.slices[static_cast<int>(s)];
  if (quotes.empty()) {
    state.SkipWithError("empty slice");
    return;
  }
  // Correctness check, once, outside the timed loop.
  std::size_t iterations = 0;
  for (const auto& o : quotes) {
    try {
      const auto res =
          solve(o.type, o.price, o.S, o.K, o.T, m.spec.r, m.spec.q);
      // Recovered price within a hundredth of a tick, or vol within 1e-6.
      const double repriced = p4mfe::ch03::BsPrice(
          o.type, o.S, o.K, o.T, res.sigma, m.spec.r, m.spec.q);
      if (std::abs(res.sigma - o.sigma) > 1e-6 &&
          std::abs(repriced - o.price) > 1e-2 * m.spec.tick) {
        state.SkipWithError("implied vol does not recover the generating vol");
        return;
      }
      iterations += res.iterations;
    } catch (const std::exception& e) {
      state.SkipWithError(std::string("implied vol threw: ") + e.what());
      return;
    }
  }
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : quotes)
      sum += solve(o.type, o.price, o.S, o.K, o.T, m.spec.r, m.spec.q).sigma;
    benchmark::DoNotOptimize(sum);
  }
  SetItemsOptions(state, quotes.size());
  state.counters["mean_iterations"] =
      static_cast<double>(iterations) / static_cast<double>(quotes.size());
}

const bool kRegistered = [] {
  (void)GetMarket();
  benchmark::RegisterBenchmark("W1/ch03/bs_price", BmBsPrice);
  benchmark::RegisterBenchmark("W1/ch03/bs_greeks", BmBsGreeks);
  benchmark::RegisterBenchmark("W1/ch01/parity_violation", BmParityViolation);
  for (const Slice s :
       {Slice::kAll, Slice::kAtm, Slice::kWings, Slice::kShort}) {
    benchmark::RegisterBenchmark(
        std::string("W1/ch03/implied_vol_newton/") + SliceName(s), BmImpliedVol,
        &IvNewton, s);
    benchmark::RegisterBenchmark(
        std::string("W1/ch03/implied_vol_bisection/") + SliceName(s),
        BmImpliedVol, &IvBisection, s);
  }
  return true;
}();

}  // namespace
