// Remaining chapter 4 paths (coverage sweep): the lognormal asset model over
// the option chain (moments, quantiles, confidence intervals, in-the-money
// probabilities), density transforms, and sums/products of independent
// variables.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/ch04/asset_model.hpp>
#include <p4mfe/ch04/change_of_variables.hpp>
#include <p4mfe/ch04/in_the_money.hpp>
#include <p4mfe/ch04/independent.hpp>
#include <p4mfe/ch04/lognormal.hpp>
#include <vector>

namespace {

namespace ch03 = p4mfe::ch03;
namespace ch04 = p4mfe::ch04;
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;

void BmAssetModel(benchmark::State& state) {
  // Real-world drift 8%; one evaluation of every model quantity per option.
  const auto spec = p4mfe::bench::ChainSpecForTier();
  const auto chain = p4mfe::bench::MakeOptionChain(spec);
  const double mu = 0.08;
  {
    const auto& o = chain.front();
    const auto ci = ch04::PriceConfidenceInterval(0.95, o.spot, o.maturity,
                                                  o.sigma, mu, spec.q);
    const double inside =
        ch04::PriceCdf(ci.hi, o.spot, o.maturity, o.sigma, mu, spec.q) -
        ch04::PriceCdf(ci.lo, o.spot, o.maturity, o.sigma, mu, spec.q);
    if (!Check(state, std::abs(inside - 0.95) < 1e-9,
               "confidence interval misses 95%"))
      return;
  }
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& o : chain) {
      const auto lr =
          ch04::LogReturnDistribution(o.maturity, o.sigma, mu, spec.q);
      sum += lr.mean + ch04::PriceMean(o.spot, o.maturity, mu, spec.q);
      sum += ch04::PriceVariance(o.spot, o.maturity, o.sigma, mu, spec.q);
      sum += ch04::PriceMedian(o.spot, o.maturity, o.sigma, mu, spec.q);
      sum += ch04::PricePdf(o.strike, o.spot, o.maturity, o.sigma, mu, spec.q);
      sum += ch04::PriceCdf(o.strike, o.spot, o.maturity, o.sigma, mu, spec.q);
      sum += ch04::PriceQuantile(0.25, o.spot, o.maturity, o.sigma, mu, spec.q);
      const auto ci = ch04::PriceConfidenceInterval(0.9, o.spot, o.maturity,
                                                    o.sigma, mu, spec.q);
      sum += ci.hi - ci.lo;
      sum += ch04::ProbItm(o.type, o.spot, o.strike, o.maturity, o.sigma, mu,
                           spec.q);
      sum += ch04::ProbBetween(0.9 * o.strike, 1.1 * o.strike, o.spot,
                               o.maturity, o.sigma, mu, spec.q);
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, chain.size(), "options");
}

void BmDensityTransforms(benchmark::State& state) {
  // Densities of aX+b, e^X, X^2 and generic transforms on a 10,000-point grid.
  const auto phi = [](double x) { return ch03::NormalPdf(x, 0.0, 1.0); };
  const auto lin = ch04::LinearTransformPdf(phi, 2.0, 1.0);
  const auto ex = ch04::ExpTransformPdf(phi);
  const auto sq = ch04::SquareTransformPdf(phi);
  const auto gen = ch04::TransformedPdf(
      phi, [](double y) { return std::log(y); },
      [](double y) { return 1.0 / y; });
  const auto gen_r = ch04::TransformedPdf(
      phi, [](double y) { return std::log(y); },
      [](double y) { return 1.0 / y; }, [](double y) { return y > 0.0; });
  const auto num = ch04::TransformedPdfNumeric(
      phi, [](double y) { return std::log(y); },
      [](double y) { return y > 0.0; });
  const std::size_t n = 10'000;
  if (!Check(state,
             std::abs(ex(1.5) - ch03::LognormalPdf(1.5, 0.0, 1.0)) < 1e-12 &&
                 std::abs(num(1.5) - ex(1.5)) < 1e-6,
             "transformed densities disagree"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 1; i <= n; ++i) {
      const double y = 5.0 * static_cast<double>(i) / static_cast<double>(n);
      sum += lin(y) + ex(y) + sq(y) + gen(y) + gen_r(y) + num(y);
    }
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, n, "points");
}

void BmIndependent(benchmark::State& state) {
  // 500 independent normals; one convolution density point; lognormal helpers.
  const std::size_t n = 500;
  std::vector<double> a(n);
  std::vector<ch03::NormalParams> x_dist(n);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = 1.0 / static_cast<double>(n);
    x_dist[i] = {0.0005 * static_cast<double>(i % 7),
                 0.0001 * static_cast<double>(1 + i % 5)};
  }
  const auto phi = [](double x) { return ch03::NormalPdf(x, 0.0, 1.0); };
  const auto joint = ch04::IndependentJointPdf(phi, phi);
  // N(0,1) + N(0,1) = N(0,2).
  const double conv = ch04::SumPdf(phi, phi, 0.5);
  if (!Check(state,
             std::abs(conv - ch03::NormalPdf(0.5, 0.0, std::sqrt(2.0))) < 1e-8,
             "sum_pdf is off"))
    return;
  for (auto _ : state) {
    const auto c = ch04::IndependentNormalCombination(a, x_dist);
    const auto s = ch04::IndependentNormalSum(x_dist);
    const auto p = ch04::IndependentLognormalProduct(a, x_dist);
    const auto p1 = ch04::IndependentLognormalProduct(x_dist);
    double sum = c.mean + s.variance + p.mean + p1.variance + joint(0.1, -0.2);
    sum += ch04::SumPdf(phi, phi, 0.5);
    sum += ch04::LognormalMoment(3.0, 0.05, 0.2) +
           ch04::LognormalQuantile(0.9, 0.05, 0.2);
    sum += ch04::LognormalMedian(0.05) + ch04::LognormalMode(0.05, 0.2);
    sum += ch04::LognormalParamsFromMoments(1.05, 0.04).variance;
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, n, "variables");
}

const bool kRegistered = [] {
  benchmark::RegisterBenchmark("W2/ch04/asset_model", BmAssetModel);
  benchmark::RegisterBenchmark("W2/ch04/density_transforms",
                               BmDensityTransforms);
  benchmark::RegisterBenchmark("W2/ch04/independent+lognormal", BmIndependent)
      ->Unit(benchmark::kMicrosecond);
  return true;
}();

}  // namespace
