// Workload W8 (docs/performance-testing-plan.md), chapter 10 part:
// polynomial interpolation on Chebyshev nodes (Lagrange form is O(n^2) per
// point, Newton form O(n) after O(n^2) setup), power series algebra, and the
// cheap appendix helpers for coverage.
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch10/interpolation.hpp>
#include <p4mfe/ch10/linear_recursions.hpp>
#include <p4mfe/ch10/power_series.hpp>
#include <p4mfe/ch10/stirling.hpp>
#include <p4mfe/ch10/sums.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace {

namespace ch10 = p4mfe::ch10;
using p4mfe::bench::check;
using p4mfe::bench::set_items;

struct Nodes {
    std::vector<double> xs, ys, eval;
};

// Chebyshev nodes on [-1, 1] for f(x) = 1 / (1 + 25 x^2) (Runge's function),
// evaluated at 10,000 points.
Nodes chebyshev(std::size_t n) {
    Nodes s;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = std::cos((2.0 * static_cast<double>(i) + 1.0) * std::numbers::pi / (2.0 * static_cast<double>(n)));
        s.xs.push_back(x);
        s.ys.push_back(1.0 / (1.0 + 25.0 * x * x));
    }
    const std::size_t m = p4mfe::bench::by_tier<std::size_t>(1'000, 10'000, 100'000);
    for (std::size_t i = 0; i < m; ++i) s.eval.push_back(-1.0 + 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(m));
    return s;
}

void BM_lagrange_interpolate(benchmark::State& state) {
    const auto s = chebyshev(static_cast<std::size_t>(state.range(0)));
    for (auto _ : state) {
        double sum = 0.0;
        for (const double x : s.eval) sum += ch10::lagrange_interpolate(s.xs, s.ys, x);
        benchmark::DoNotOptimize(sum);
    }
    set_items(state, s.eval.size(), "points");
    state.SetComplexityN(state.range(0));
}

void BM_newton_interpolate(benchmark::State& state) {
    const auto s = chebyshev(static_cast<std::size_t>(state.range(0)));
    const auto c = ch10::divided_differences(s.xs, s.ys);
    double worst = 0.0;
    for (const double x : s.eval)
        worst = std::max(worst, std::abs(ch10::newton_interpolate(s.xs, c, x) - ch10::lagrange_interpolate(s.xs, s.ys, x)));
    // Divided differences lose accuracy as n grows (Chebyshev nodes in their
    // natural order); the gap to the Lagrange form is reported, and only a
    // gross disagreement fails.
    if (!check(state, worst < 1e-2, "Newton and Lagrange forms disagree")) return;
    for (auto _ : state) {
        const auto cc = ch10::divided_differences(s.xs, s.ys);
        double sum = 0.0;
        for (const double x : s.eval) sum += ch10::newton_interpolate(s.xs, cc, x);
        benchmark::DoNotOptimize(sum);
    }
    set_items(state, s.eval.size(), "points");
    state.counters["max_gap_to_lagrange"] = worst;
}

void BM_interpolating_polynomial(benchmark::State& state) {
    const auto s = chebyshev(static_cast<std::size_t>(state.range(0)));
    for (auto _ : state) {
        const auto a = ch10::interpolating_polynomial(s.xs, s.ys);
        benchmark::DoNotOptimize(ch10::polynomial_value(a, 0.3));
    }
    state.SetComplexityN(state.range(0));
}

void BM_power_series(benchmark::State& state) {
    const auto n = static_cast<unsigned>(state.range(0));
    for (auto _ : state) {
        const auto e = ch10::exp_power_series(n);
        const auto l = ch10::log1p_power_series(n);
        const auto p = ch10::cauchy_product(e, l);
        const auto d = ch10::derivative(p);
        auto a = ch10::antiderivative(d);
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(ch10::radius_ratio_estimate(l.a));
        benchmark::DoNotOptimize(ch10::radius_root_estimate(l.a));
    }
    state.SetComplexityN(state.range(0));
}

void BM_appendix_helpers(benchmark::State& state) {
    const std::vector<double> a{1.0, 1.0}, init{0.0, 1.0};
    for (auto _ : state) {
        benchmark::DoNotOptimize(ch10::recursion_terms(a, init, 90));
        benchmark::DoNotOptimize(ch10::solve_second_order(1.0, 1.0, 0.0, 1.0));
        benchmark::DoNotOptimize(ch10::loan_payment(300'000.0, 0.06 / 12.0, 360));
        benchmark::DoNotOptimize(ch10::sums_of_powers_recursive(1000, 6));
        benchmark::DoNotOptimize(ch10::sum_of_powers_polynomial(8));
        benchmark::DoNotOptimize(ch10::sum_kj_xk_infinite(4, 0.5));
        benchmark::DoNotOptimize(ch10::log_stirling(1000.0, 4));
        benchmark::DoNotOptimize(ch10::central_binomial_stirling(500.0));
    }
}

const bool registered = [] {
    const auto ns = p4mfe::bench::by_tier<std::vector<std::int64_t>>({10, 50}, {10, 50, 200}, {50, 200, 1000});
    auto* l = benchmark::RegisterBenchmark("W8/ch10/lagrange_interpolate", BM_lagrange_interpolate);
    auto* n = benchmark::RegisterBenchmark("W8/ch10/divided_differences+newton_interpolate", BM_newton_interpolate);
    auto* p = benchmark::RegisterBenchmark("W8/ch10/interpolating_polynomial", BM_interpolating_polynomial);
    for (const auto k : ns) {
        l->Arg(k);
        p->Arg(k);
    }
    // The Newton form with nodes in natural order is unusable beyond ~50
    // Chebyshev nodes (gap to the Lagrange form > 1e-2 at 200), so it stops at 50.
    for (const auto k : {10, 50}) n->Arg(k);
    l->Complexity(benchmark::oNSquared)->Unit(benchmark::kMillisecond);
    n->Unit(benchmark::kMillisecond);
    p->Complexity(benchmark::oNSquared)->Unit(benchmark::kMicrosecond);
    auto* ps = benchmark::RegisterBenchmark("W8/ch10/power_series", BM_power_series);
    for (const auto k : p4mfe::bench::by_tier<std::vector<std::int64_t>>({100}, {100, 1'000, 10'000}, {1'000, 10'000, 30'000}))
        ps->Arg(k);
    ps->Complexity(benchmark::oNSquared)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("W8/ch10/appendix_helpers", BM_appendix_helpers)->Unit(benchmark::kMicrosecond);
    return true;
}();

}  // namespace
