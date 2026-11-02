// Remaining chapter 5 path (coverage sweep): the estimated order of
// convergence over Newton's iterates for sqrt(c), c across 1,000 values.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch05/solvers.hpp>

namespace {

namespace ch05 = p4mfe::ch05;
using p4mfe::bench::Check;
using p4mfe::bench::SetItems;

double NewtonOrder(double c) {
  const auto res = ch05::Newton([c](double x) { return x * x - c; },
                                [](double x) { return 2.0 * x; }, 1.0 + c,
                                {1e-10, 1e-13, 100});
  const double root = std::sqrt(c);
  // The last three iterates whose errors are still above round-off.
  double best = 0.0;
  const auto& it = res.iterates;
  for (std::size_t k = 2; k < it.size(); ++k) {
    const double e0 = std::abs(it[k - 2] - root),
                 e1 = std::abs(it[k - 1] - root), e2 = std::abs(it[k] - root);
    if (e2 > 1e-12 && e1 < 1e-1) best = ch05::ConvergenceOrder(e0, e1, e2);
  }
  return best;
}

void BmConvergenceOrder(benchmark::State& state) {
  const std::size_t n = 1000;
  if (!Check(state, std::abs(NewtonOrder(2.0) - 2.0) < 0.3,
             "Newton's order is not about 2"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 1; i <= n; ++i)
      sum += NewtonOrder(static_cast<double>(i));
    benchmark::DoNotOptimize(sum);
  }
  SetItems(state, n, "solves");
}

const bool kRegistered = [] {
  benchmark::RegisterBenchmark("W8/ch05/convergence_order", BmConvergenceOrder)
      ->Unit(benchmark::kMicrosecond);
  return true;
}();

}  // namespace
