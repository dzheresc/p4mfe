// Chapter 5 benchmarks:
//   W1  implied vol by Newton, secant and bisection (Table 5.7) over the chain
//   W3  bond yields (Table 5.6) over the bond universe; bootstrapping with
//       linear interpolation over benchmark bonds and dense strips
//   W7  Gaussian elimination and N-dimensional Newton (Tables 5.4, 5.5)
//
// Every scalar solver returns RootResult, which records all iterates in a
// std::vector, so allocations per call are part of what is measured here.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/bonds.hpp>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch05/bond_yield.hpp>
#include <p4mfe/ch05/bootstrapping.hpp>
#include <p4mfe/ch05/implied_vol.hpp>
#include <p4mfe/ch05/newton_nd.hpp>
#include <random>
#include <string>
#include <vector>

namespace {

namespace ch05 = p4mfe::ch05;
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;
using p4mfe::ch01::OptionType;

// ----------------------------------------------------------------- W1 -----

const std::vector<p4mfe::bench::OptionQuote>& Chain() {
  static const auto kC =
      p4mfe::bench::MakeOptionChain(p4mfe::bench::ChainSpecForTier());
  return kC;
}

using IvSolver = ch05::RootResult (*)(OptionType, double, double, double,
                                      double, double, double);
ch05::RootResult IvNewton(OptionType t, double p, double spot, double strike,
                          double maturity, double r, double q) {
  return ch05::ImpliedVolNewton(t, p, spot, strike, maturity, r, q);
}
ch05::RootResult IvSecant(OptionType t, double p, double spot, double strike,
                          double maturity, double r, double q) {
  return ch05::ImpliedVolSecant(t, p, spot, strike, maturity, r, q);
}
ch05::RootResult IvBisection(OptionType t, double p, double spot, double strike,
                             double maturity, double r, double q) {
  return ch05::ImpliedVolBisection(t, p, spot, strike, maturity, r, q);
}

void BmImpliedVol(benchmark::State& state, IvSolver solve) {
  const auto& c = Chain();
  const p4mfe::bench::ChainSpec spec;
  std::size_t iterations = 0, failures = 0;
  for (const auto& o : c) {
    try {
      const auto res =
          solve(o.type, o.price, o.spot, o.strike, o.maturity, spec.r, spec.q);
      iterations += res.iterations;
      const double repriced = p4mfe::ch03::BsPrice(
          o.type, o.spot, o.strike, o.maturity, res.x, spec.r, spec.q);
      if (std::abs(res.x - o.sigma) > 1e-5 &&
          std::abs(repriced - o.price) > 1e-3)
        ++failures;
    } catch (const std::exception&) {
      ++failures;
    }
  }
  if (!Check(state, failures == 0,
             std::to_string(failures) + " quotes did not recover their vol"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : c)
      sum +=
          solve(o.type, o.price, o.spot, o.strike, o.maturity, spec.r, spec.q)
              .x;
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, c.size(), "options");
  state.counters["mean_iterations"] =
      static_cast<double>(iterations) / static_cast<double>(c.size());
}

// ----------------------------------------------------------------- W3 -----

const std::vector<p4mfe::bench::Bond>& Universe() {
  static const auto kU =
      p4mfe::bench::BondUniverse(p4mfe::bench::BondUniverseSize());
  return kU;
}

template <class Solve>
void BmBondYield(benchmark::State& state, Solve solve) {
  const auto& u = Universe();
  double worst = 0.0;
  for (const auto& b : u)
    worst = std::max(
        worst, std::abs(p4mfe::ch02::BondPrice(b.cf, solve(b).x) - b.price));
  if (!Check(state, worst < 1e-6, "bond yield does not reprice")) return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) sum += solve(b).x;
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, u.size(), "bonds");
}

void BmBootstrap(benchmark::State& state,
                 std::vector<p4mfe::ch02::BondQuote> quotes, int freq,
                 double tol) {
  const auto curve = ch05::BootstrapZeroCurve(quotes, freq);
  double worst = 0.0;
  for (std::size_t i = 0; i < curve.times.size(); ++i)
    worst = std::max(
        worst,
        std::abs(curve.rates[i] - p4mfe::bench::TrueZeroRate(curve.times[i])));
  if (!Check(state, worst < tol, "bootstrap is too far from the true curve"))
    return;
  for (auto _ : state)
    benchmark::DoNotOptimize(ch05::BootstrapZeroCurve(quotes, freq));
  SetItems(state, quotes.size(), "bonds");
  state.counters["max_rate_error"] = worst;
}

void BmZeroRateLinear(benchmark::State& state) {
  const auto curve =
      ch05::BootstrapZeroCurve(p4mfe::bench::ParBondStrip(30.0, 2), 2);
  std::vector<double> ts(10'000);
  for (std::size_t i = 0; i < ts.size(); ++i)
    ts[i] = 30.0 * static_cast<double>(i) / static_cast<double>(ts.size());
  for (auto _ : state) {
    double sum = 0.0;
    for (const double t : ts) sum += ch05::ZeroRateLinear(curve, t);
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, ts.size(), "lookups");
}

// ----------------------------------------------------------------- W7 -----

// Diagonally dominant random system, so it is well conditioned at any N.
struct LinearSystem {
  ch05::Matrix a{0, 0};
  ch05::Vector b;
};
LinearSystem RandomSystem(std::size_t n) {
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  LinearSystem s{ch05::Matrix(n, n), ch05::Vector(n)};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) s.a(i, j) = u(rng);
    s.a(i, i) += static_cast<double>(n);
    s.b[i] = u(rng);
  }
  return s;
}

void BmSolveLinear(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto sys = RandomSystem(n);
  const auto x = ch05::SolveLinear(sys.a, sys.b);
  double res = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double s = -sys.b[i];
    for (std::size_t j = 0; j < n; ++j) s += sys.a(i, j) * x[j];
    res = std::max(res, std::abs(s));
  }
  if (!Check(state, res < 1e-10, "solve_linear residual too large")) return;
  for (auto _ : state)
    benchmark::DoNotOptimize(ch05::SolveLinear(sys.a, sys.b));
  state.SetComplexityN(static_cast<std::int64_t>(n));
}

// F_i(x) = x_i^3 + sum_j a_ij x_j - b_i, a well-conditioned nonlinear system
// with a known root x* = 0.5 (b is built from it).
struct NlSystem {
  ch05::Matrix a{0, 0};
  ch05::Vector b;
};
NlSystem NonlinearSystem(std::size_t n) {
  auto s = RandomSystem(n);
  NlSystem out{s.a, ch05::Vector(n)};
  for (std::size_t i = 0; i < n; ++i) {
    double v = 0.125;
    for (std::size_t j = 0; j < n; ++j) v += s.a(i, j) * 0.5;
    out.b[i] = v;
  }
  return out;
}

void BmNewtonNd(benchmark::State& state, bool approximate) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto sys = NonlinearSystem(n);
  std::size_t evals = 0;
  const auto func = [&](const ch05::Vector& x) {
    ++evals;
    ch05::Vector f(n);
    for (std::size_t i = 0; i < n; ++i) {
      double v = x[i] * x[i] * x[i] - sys.b[i];
      for (std::size_t j = 0; j < n; ++j) v += sys.a(i, j) * x[j];
      f[i] = v;
    }
    return f;
  };
  const auto jacobian = [&](const ch05::Vector& x) {
    ch05::Matrix m = sys.a;
    for (std::size_t i = 0; i < n; ++i) m(i, i) += 3.0 * x[i] * x[i];
    return m;
  };
  const ch05::Vector x0(n, 1.0);
  const auto solve = [&] {
    return approximate ? ch05::ApproximateNewtonNd(func, x0)
                       : ch05::NewtonNd(func, jacobian, x0);
  };
  const auto res = solve();
  double err = 0.0;
  for (const double v : res.x) err = std::max(err, std::abs(v - 0.5));
  if (!Check(state, err < 1e-6, "Newton did not find the root")) return;
  const std::size_t evals_per_solve = evals;
  for (auto _ : state) benchmark::DoNotOptimize(solve());
  state.counters["iterations"] = static_cast<double>(res.iterations);
  state.counters["F_evals"] = static_cast<double>(evals_per_solve);
  state.SetComplexityN(static_cast<std::int64_t>(n));
}

const bool kRegistered = [] {
  (void)Chain();
  (void)Universe();
  benchmark::RegisterBenchmark("W1/ch05/implied_vol_newton", BmImpliedVol,
                               &IvNewton);
  benchmark::RegisterBenchmark("W1/ch05/implied_vol_secant", BmImpliedVol,
                               &IvSecant);
  benchmark::RegisterBenchmark("W1/ch05/implied_vol_bisection", BmImpliedVol,
                               &IvBisection);

  using Bond = p4mfe::bench::Bond;
  benchmark::RegisterBenchmark(
      "W3/ch05/bond_yield_newton",
      BmBondYield<ch05::RootResult (*)(const Bond&)>,
      [](const Bond& b) { return ch05::BondYieldNewton(b.cf, b.price); });
  benchmark::RegisterBenchmark(
      "W3/ch05/bond_yield_secant",
      BmBondYield<ch05::RootResult (*)(const Bond&)>,
      [](const Bond& b) { return ch05::BondYieldSecant(b.cf, b.price); });
  benchmark::RegisterBenchmark(
      "W3/ch05/bond_yield_bisection",
      BmBondYield<ch05::RootResult (*)(const Bond&)>,
      [](const Bond& b) { return ch05::BondYieldBisection(b.cf, b.price); });
  // Interpolation error between sparse benchmark maturities is real, so the
  // check there is loose; the dense strip should match closely.
  benchmark::RegisterBenchmark("W3/ch05/bootstrap_zero_curve/benchmark_bonds",
                               BmBootstrap, p4mfe::bench::BenchmarkBonds(2), 2,
                               2e-3);
  benchmark::RegisterBenchmark("W3/ch05/bootstrap_zero_curve/strip_30y",
                               BmBootstrap, p4mfe::bench::ParBondStrip(30.0, 2),
                               2, 1e-9);
  benchmark::RegisterBenchmark("W3/ch05/zero_rate_linear", BmZeroRateLinear);

  auto* lin =
      benchmark::RegisterBenchmark("W7/ch05/solve_linear", BmSolveLinear);
  for (const auto n : p4mfe::bench::ByTier<std::vector<std::int64_t>>(
           {10, 50}, {10, 50, 100, 250, 500}, {100, 250, 500, 1000}))
    lin->Arg(n);
  lin->Complexity(benchmark::oNCubed)->Unit(benchmark::kMicrosecond);
  const auto nl = p4mfe::bench::ByTier<std::vector<std::int64_t>>(
      {2, 10}, {2, 10, 50, 100}, {10, 50, 100, 200});
  for (const bool approx : {false, true}) {
    auto* b = benchmark::RegisterBenchmark(
        approx ? "W7/ch05/approximate_newton_nd" : "W7/ch05/newton_nd",
        BmNewtonNd, approx);
    for (const auto n : nl) b->Arg(n);
    b->Unit(benchmark::kMicrosecond);
  }
  return true;
}();

}  // namespace
