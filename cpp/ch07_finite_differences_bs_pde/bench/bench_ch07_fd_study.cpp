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
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;

void BmFdStudy(benchmark::State& state) {
  const auto f = [](double x) { return std::exp(x) * std::sin(x); };
  const auto df = [](double x) {
    return std::exp(x) * (std::sin(x) + std::cos(x));
  };
  const double p =
      ch07::FdObservedOrder(f, 0.5, df(0.5), ch07::FdScheme::kCentral, 1e-2);
  if (!Check(
          state,
          std::abs(p - ch07::TruncationOrder(ch07::FdScheme::kCentral)) < 0.1,
          "central order is not 2"))
    return;
  const std::size_t n = 1000;
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double a =
          -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(n);
      for (const auto s : {ch07::FdScheme::kForward, ch07::FdScheme::kBackward,
                           ch07::FdScheme::kCentral}) {
        const auto rows = ch07::FdErrorTable(f, a, df(a), s, 0.1, 8);
        sum += rows.back().error + ch07::FdObservedOrder(f, a, df(a), s, 1e-3);
        sum += ch07::OptimalStep(s, 1, a) + ch07::TruncationOrder(s);
      }
      sum += ch07::BackwardDifference(
          f, a, ch07::OptimalStep(ch07::FdScheme::kBackward, 2, a));
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, n, "points");
}

void BmPdeSolutions(benchmark::State& state) {
  const auto spec = p4mfe::bench::ChainSpecForTier();
  const auto chain = p4mfe::bench::MakeOptionChain(spec);
  std::vector<ch03::Greeks> gs;
  std::vector<double> w;
  for (const auto& o : chain) {
    gs.push_back(ch03::BsGreeks(o.type, o.spot, o.strike, o.maturity, o.sigma,
                                spec.r, spec.q));
    w.push_back(o.strike > o.spot ? 1.0 : -1.0);
  }
  const auto& o0 = chain.front();
  const auto call = [&](double spot, double t) {
    return ch03::BsPrice(o0.type, spot, o0.strike, o0.maturity - t, o0.sigma,
                         spec.r, spec.q);
  };
  if (!Check(state,
             std::abs(ch07::BsPdeResidualFd(call, o0.spot, 0.0, o0.sigma,
                                            spec.r, spec.q)) < 1e-2,
             "Black-Scholes price does not solve the PDE"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : chain) {
      sum +=
          ch07::PowerSolution(2.0, o.spot, 0.0, o.sigma, spec.r, spec.q) * 1e-8;
      sum += ch07::PowerSolutionRate(-1.0, o.sigma, spec.r, spec.q);
      sum += ch07::ForwardContractValue(o.spot, 0.0, o.strike, o.maturity,
                                        spec.r, spec.q);
    }
    sum += ch07::BsPdeResidualFd(call, o0.spot, 0.0, o0.sigma, spec.r, spec.q);
    const auto g = ch07::CombineGreeks(w, gs);
    benchmark::DoNotOptimize(sum + g.delta + g.gamma);
  }
  SetItems(state, chain.size(), "options");
}

const bool kRegistered = [] {
  benchmark::RegisterBenchmark("W4/ch07/fd_accuracy_study", BmFdStudy)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W1/ch07/pde_closed_forms+combine_greeks",
                               BmPdeSolutions)
      ->Unit(benchmark::kMicrosecond);
  return true;
}();

}  // namespace
