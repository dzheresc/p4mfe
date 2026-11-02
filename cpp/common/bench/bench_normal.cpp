// Workload W2 (docs/performance-testing-plan.md): the normal distribution
// kernels everything else is built on. Points are spread over [-8, 8]
// (probabilities over (1e-12, 1 - 1e-12) for the inverse), so both tails and
// the centre are exercised. Reported as evaluations/second.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/normal.hpp>
#include <vector>

namespace {

using p4mfe::bench::set_items;

struct Points {
  std::vector<double> x;  // in [-8, 8]
  std::vector<double> p;  // in (1e-12, 1 - 1e-12), half of them in the tails
};

const Points& GetPoints() {
  static const Points kPts = [] {
    const std::size_t n =
        p4mfe::bench::by_tier<std::size_t>(10'000, 1'000'000, 10'000'000);
    Points out{std::vector<double>(n), std::vector<double>(n)};
    for (std::size_t i = 0; i < n; ++i) {
      const double u = (static_cast<double>(i) + 0.5) / static_cast<double>(n);
      out.x[i] = -8.0 + 16.0 * u;
      out.p[i] = p4mfe::NormCdf(-7.0 + 14.0 * u);  // ~1e-12 .. 1 - 1e-12
    }
    return out;
  }();
  return kPts;
}

template <double (*F)(double) noexcept>
void BmKernel(benchmark::State& state, bool probabilities) {
  const auto& v = probabilities ? GetPoints().p : GetPoints().x;
  for (auto _ : state) {
    double sum = 0.0;
    for (const double x : v) sum += F(x);
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, v.size(), "points");
}

// Round trip x -> N(x) -> N^-1, to check NormInv in setup. Beyond |x| = 5
// the upper tail of N(x) has too few significant digits for a tight check.
void BmNormInvCheck(benchmark::State& state) {
  double worst = 0.0;
  for (const double x : GetPoints().x)
    if (std::abs(x) < 5.0)
      worst = std::max(worst, std::abs(p4mfe::NormInv(p4mfe::NormCdf(x)) - x));
  if (!p4mfe::bench::check(state, worst < 1e-6,
                           "NormInv(NormCdf(x)) does not round trip"))
    return;
  BmKernel<p4mfe::NormInv>(state, true);
}

const bool kRegistered = [] {
  (void)GetPoints();
  benchmark::RegisterBenchmark("W2/common/norm_pdf", BmKernel<p4mfe::NormPdf>,
                               false);
  benchmark::RegisterBenchmark("W2/common/norm_cdf", BmKernel<p4mfe::NormCdf>,
                               false);
  benchmark::RegisterBenchmark("W2/common/norm_inv", BmNormInvCheck);
  return true;
}();

}  // namespace
