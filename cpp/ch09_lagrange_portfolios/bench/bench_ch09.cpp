// Workload W7 (docs/performance-testing-plan.md), chapter 9 part: mean-variance
// portfolios over an N-asset factor-model universe (N up to an S&P 500 sized
// universe), and the general Lagrange multiplier solver on the same problem.
#include <p4mfe/bench/portfolio.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch09/lagrange.hpp>
#include <p4mfe/ch09/portfolio.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

namespace ch09 = p4mfe::ch09;
using p4mfe::bench::check;
using p4mfe::bench::set_items;

void BM_frontier(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto u = p4mfe::bench::asset_universe(n);
    const auto gmv = ch09::global_min_variance_portfolio(u.Sigma);
    double sum_w = 0.0;
    for (const double w : gmv) sum_w += w;
    if (!check(state, std::abs(sum_w - 1.0) < 1e-9, "minimum variance weights do not sum to 1")) return;
    for (auto _ : state) {
        const auto c = ch09::frontier_constants(u.mu, u.Sigma);
        benchmark::DoNotOptimize(c);
        benchmark::DoNotOptimize(ch09::global_min_variance_portfolio(u.Sigma));
        benchmark::DoNotOptimize(ch09::tangency_portfolio(u.mu, u.Sigma, 0.03));
    }
    state.SetComplexityN(static_cast<std::int64_t>(n));
}

// The efficient frontier traced at 100 target returns: each point solves the
// covariance system again, as the API does.
void BM_frontier_trace(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto u = p4mfe::bench::asset_universe(n);
    for (auto _ : state) {
        double sum = 0.0;
        for (int k = 0; k < 100; ++k) {
            const double target = 0.03 + 0.001 * k;
            const auto w = ch09::min_variance_portfolio(u.mu, u.Sigma, target);
            sum += ch09::portfolio_variance(w, u.Sigma);
        }
        benchmark::DoNotOptimize(sum);
    }
    set_items(state, 100, "points");
    state.SetComplexityN(static_cast<std::int64_t>(n));
}

void BM_portfolio_variance(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto u = p4mfe::bench::asset_universe(n);
    const std::vector<double> w(n, 1.0 / static_cast<double>(n));
    for (auto _ : state) benchmark::DoNotOptimize(ch09::portfolio_variance(w, u.Sigma));
    state.SetComplexityN(static_cast<std::int64_t>(n));
}

// min w' Sigma w subject to sum w = 1 and mu' w = target, by the general
// solver (finite-difference gradients and Hessians), against the closed form.
void BM_lagrange(benchmark::State& state) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto u = p4mfe::bench::asset_universe(n);
    const double target = 0.05;
    const auto f = [&](const ch09::Vector& w) { return ch09::portfolio_variance(w, u.Sigma); };
    const auto g = [&](const ch09::Vector& w) {
        double s = -1.0;
        for (const double x : w) s += x;
        return ch09::Vector{s, ch09::dot(u.mu, w) - target};
    };
    const ch09::Vector w0(n, 1.0 / static_cast<double>(n));
    const auto res = ch09::lagrange_stationary_point(f, g, w0);
    const auto exact = ch09::min_variance_portfolio(u.mu, u.Sigma, target);
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(res.x[i] - exact[i]));
    if (!check(state, err < 1e-4, "Lagrange solver disagrees with the closed form")) return;
    for (auto _ : state) {
        const auto r = ch09::lagrange_stationary_point(f, g, w0);
        benchmark::DoNotOptimize(ch09::classify_constrained(f, g, r.x, r.lambda));
    }
    state.counters["iterations"] = static_cast<double>(res.iterations);
    state.SetComplexityN(static_cast<std::int64_t>(n));
}

const bool registered = [] {
    const auto ns = p4mfe::bench::by_tier<std::vector<std::int64_t>>({10, 50}, {10, 100, 500}, {100, 500, 1000});
    auto* f = benchmark::RegisterBenchmark("W7/ch09/frontier+gmv+tangency", BM_frontier);
    auto* t = benchmark::RegisterBenchmark("W7/ch09/frontier_trace_100", BM_frontier_trace);
    auto* v = benchmark::RegisterBenchmark("W7/ch09/portfolio_variance", BM_portfolio_variance);
    for (const auto n : ns) {
        f->Arg(n);
        t->Arg(n);
        v->Arg(n);
    }
    f->Complexity(benchmark::oNCubed)->Unit(benchmark::kMillisecond);
    t->Complexity(benchmark::oNCubed)->Unit(benchmark::kMillisecond);
    v->Complexity(benchmark::oNSquared)->Unit(benchmark::kMicrosecond);
    auto* l = benchmark::RegisterBenchmark("W7/ch09/lagrange_stationary_point", BM_lagrange);
    for (const auto n : p4mfe::bench::by_tier<std::vector<std::int64_t>>({3}, {3, 10, 30}, {10, 30, 60})) l->Arg(n);
    l->Unit(benchmark::kMillisecond);
    return true;
}();

}  // namespace
