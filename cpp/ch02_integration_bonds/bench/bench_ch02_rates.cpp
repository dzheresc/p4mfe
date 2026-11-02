// Remaining chapter 2 paths (coverage sweep): rate conversions, forward rates
// and discount factors on the true curve (closed form and from the
// instantaneous rate by quadrature), zero coupon bonds, duration-based price
// approximations, observed convergence order, and differentiating an
// improper integral.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/bonds.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch02/rates.hpp>
#include <vector>

namespace {

namespace ch02 = p4mfe::ch02;
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;

// Monthly grid out to 30 years.
std::vector<double> Grid() {
  std::vector<double> t;
  for (int m = 1; m <= 360; ++m) t.push_back(m / 12.0);
  return t;
}

void BmCurveClosedForms(benchmark::State& state) {
  const auto ts = Grid();
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 1; i < ts.size(); ++i) {
      const double t1 = ts[i - 1], t2 = ts[i];
      const double r1 = p4mfe::bench::TrueZeroRate(t1),
                   r2 = p4mfe::bench::TrueZeroRate(t2);
      const double discount1 = ch02::DiscountFactor(r1, t1),
                   discount2 = ch02::DiscountFactor(r2, t2);
      sum += ch02::ForwardRate(r1, t1, r2, t2) +
             ch02::ForwardRateFromDiscounts(discount1, t1, discount2, t2);
      sum += ch02::DiscreteFromContinuous(r2, 2.0) +
             ch02::ContinuousFromDiscrete(r2, 2.0);
      sum += ch02::InstRateFromZeroRate(p4mfe::bench::TrueZeroRate, t2);
      const double z = ch02::ZeroCouponBondPrice(t2, r2);
      sum += ch02::ZeroRateFromZeroCouponPrice(z, t2);
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, ts.size() - 1, "points");
}

// Curve quantities from the instantaneous rate: one adaptive integral each.
void BmCurveFromInstRate(benchmark::State& state) {
  std::vector<double> ts;
  for (int y = 1; y <= 30; ++y) ts.push_back(y);
  const double err =
      std::abs(ch02::ZeroRateFromInstRate(p4mfe::bench::TrueInstRate, 10.0) -
               p4mfe::bench::TrueZeroRate(10.0));
  if (!Check(state, err < 1e-9, "zero rate from instantaneous rate is off"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const double t : ts) {
      sum += ch02::DiscountFactorFromInstRate(p4mfe::bench::TrueInstRate, t);
      sum += ch02::ZeroRateFromInstRate(p4mfe::bench::TrueInstRate, t);
      sum +=
          ch02::ForwardRateFromInstRate(p4mfe::bench::TrueInstRate, t - 1.0, t);
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, ts.size(), "maturities");
}

void BmDurationApproximations(benchmark::State& state) {
  const auto u = p4mfe::bench::BondUniverse(
      p4mfe::bench::ByTier<std::size_t>(100, 10'000, 100'000));
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& b : u) {
      const double price = ch02::BondPrice(b.cf, 0.05);
      const double duration = ch02::BondDuration(b.cf, 0.05),
                   convexity = ch02::BondConvexity(b.cf, 0.05);
      sum +=
          ch02::PriceApproxDuration(price, duration, 0.001) +
          ch02::PriceApproxDurationConvexity(price, duration, convexity, 0.001);
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, u.size(), "bonds");
}

void BmIntegrationMisc(benchmark::State& state) {
  // d/dt integral_0^inf e^(-t x^2) dx = -sqrt(pi) / (4 t^1.5).
  const auto f = [](double x, double t) { return std::exp(-t * x * x); };
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        ch02::ImproperIntegralDerivative(f, 0.0, ch02::kInfinity, 1.0));
    benchmark::DoNotOptimize(
        ch02::ObservedOrder([](double x) { return std::exp(x); }, 0.0, 1.0,
                            std::exp(1.0) - 1.0, 64, ch02::Rule::kSimpson));
  }
}

const bool kRegistered = [] {
  benchmark::RegisterBenchmark("W3/ch02/curve_closed_forms",
                               BmCurveClosedForms);
  benchmark::RegisterBenchmark("W3/ch02/curve_from_inst_rate",
                               BmCurveFromInstRate)
      ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("W3/ch02/duration_approximations",
                               BmDurationApproximations);
  benchmark::RegisterBenchmark(
      "W4/ch02/improper_integral_derivative+observed_order", BmIntegrationMisc)
      ->Unit(benchmark::kMillisecond);
  return true;
}();

}  // namespace
