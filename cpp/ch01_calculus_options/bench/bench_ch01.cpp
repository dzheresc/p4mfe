// Workload W8 (docs/performance-testing-plan.md), chapter 1 part: numerical
// derivatives, Riemann sums, forwards with discrete dividends, and the
// multivariable gradient/Hessian helpers that chapters 6, 8 and 9 build on.
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch01/forwards.hpp>
#include <p4mfe/ch01/multivariable.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace {

namespace ch01 = p4mfe::ch01;
using p4mfe::bench::set_items;

void BM_derivatives(benchmark::State& state) {
    const std::size_t n = p4mfe::bench::by_tier<std::size_t>(1'000, 100'000, 1'000'000);
    const auto f = [](double x) { return std::exp(-x * x) * std::sin(3.0 * x); };
    for (auto _ : state) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double x = -3.0 + 6.0 * static_cast<double>(i) / static_cast<double>(n);
            sum += ch01::derivative(f, x) + ch01::second_derivative(f, x);
        }
        benchmark::DoNotOptimize(sum);
    }
    set_items(state, n, "points");
}

void BM_riemann(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto f = [](double x) { return std::exp(-x * x); };
    for (auto _ : state) benchmark::DoNotOptimize(ch01::riemann_sum(f, 0.0, 1.0, n));
    state.SetComplexityN(static_cast<std::int64_t>(n));
}

void BM_forward_dividends(benchmark::State& state) {
    // Quarterly dividends over 30 years on 10,000 forwards.
    std::vector<ch01::Dividend> divs;
    for (int k = 1; k <= 120; ++k) divs.push_back({0.25 * k, 1.5});
    const std::size_t n = p4mfe::bench::by_tier<std::size_t>(100, 10'000, 100'000);
    for (auto _ : state) {
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double T = 0.5 + 29.5 * static_cast<double>(i) / static_cast<double>(n);
            const auto paid = static_cast<std::size_t>(T / 0.25);  // dividends in (0, T]
            sum += ch01::forward_price(100.0, T, 0.045, std::span<const ch01::Dividend>(divs.data(), paid));
        }
        benchmark::DoNotOptimize(sum);
    }
    set_items(state, n, "forwards");
}

void BM_gradient_hessian(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto f = [](const ch01::Vector& x) {
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) s += std::cos(x[i]) * (1.0 + 0.1 * static_cast<double>(i)) + x[i] * x[(i + 1) % x.size()];
        return s;
    };
    const ch01::Vector x(n, 0.3);
    for (auto _ : state) {
        benchmark::DoNotOptimize(ch01::gradient(f, x));
        benchmark::DoNotOptimize(ch01::hessian(f, x));
    }
    state.SetComplexityN(static_cast<std::int64_t>(n));
}

const bool registered = [] {
    benchmark::RegisterBenchmark("W8/ch01/derivative+second_derivative", BM_derivatives)->Unit(benchmark::kMillisecond);
    auto* r = benchmark::RegisterBenchmark("W8/ch01/riemann_sum", BM_riemann);
    for (const auto n : p4mfe::bench::by_tier<std::vector<std::int64_t>>({1 << 10}, {1 << 10, 1 << 16, 1 << 22}, {1 << 16, 1 << 22, 1 << 24}))
        r->Arg(n);
    r->Complexity(benchmark::oN)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("W8/ch01/forward_price_dividends", BM_forward_dividends)->Unit(benchmark::kMicrosecond);
    auto* g = benchmark::RegisterBenchmark("W8/ch01/gradient+hessian", BM_gradient_hessian);
    for (const auto n : p4mfe::bench::by_tier<std::vector<std::int64_t>>({2, 10}, {2, 10, 50}, {10, 50, 100})) g->Arg(n);
    g->Complexity(benchmark::oNCubed)->Unit(benchmark::kMicrosecond);
    return true;
}();

}  // namespace
