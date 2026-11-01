// Workloads W4 and W5 (docs/performance-testing-plan.md), chapter 4 part:
// risk-neutral pricing by quadrature (call, digital, power payoff) and by
// Monte Carlo (antithetic on and off). Monte Carlo memory must stay O(1).
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch04/in_the_money.hpp>
#include <p4mfe/ch04/risk_neutral.hpp>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

namespace ch04 = p4mfe::ch04;
using p4mfe::bench::check;
using p4mfe::bench::set_items;
using p4mfe::ch01::OptionType;

constexpr double S0 = 5000.0, K = 5100.0, T = 0.5, sigma = 0.2, r = 0.045, q = 0.015;

void BM_quadrature_call(benchmark::State& state) {
    const auto call = [](double s) { return std::max(s - K, 0.0); };
    const double exact = p4mfe::ch03::bs_price(OptionType::Call, S0, K, T, sigma, r, q);
    if (!check(state, std::abs(ch04::risk_neutral_price(call, S0, T, sigma, r, q) - exact) < 1e-5 * S0,
               "quadrature call price is off"))
        return;
    for (auto _ : state) benchmark::DoNotOptimize(ch04::risk_neutral_price(call, S0, T, sigma, r, q));
}

void BM_quadrature_digital(benchmark::State& state) {
    const auto digital = [](double s) { return s > K ? 1.0 : 0.0; };
    const double exact = ch04::cash_or_nothing(OptionType::Call, S0, K, T, sigma, r, q);
    if (!check(state, std::abs(ch04::risk_neutral_price(digital, S0, T, sigma, r, q) - exact) < 1e-4,
               "quadrature digital price is off"))
        return;
    for (auto _ : state) benchmark::DoNotOptimize(ch04::risk_neutral_price(digital, S0, T, sigma, r, q));
}

void BM_quadrature_power(benchmark::State& state) {
    // E[(S_T / S0)^2] = exp((2(r - q) + sigma^2) T). Scaled by S0 so the
    // absolute tolerance of the quadrature is meaningful.
    const auto sq = [](double s) { return (s / S0) * (s / S0); };
    const double exact = std::exp((2.0 * (r - q) + sigma * sigma) * T);
    if (!check(state, std::abs(ch04::risk_neutral_expectation(sq, S0, T, sigma, r, q) - exact) < 1e-6 * exact,
               "quadrature power payoff is off"))
        return;
    for (auto _ : state) benchmark::DoNotOptimize(ch04::risk_neutral_expectation(sq, S0, T, sigma, r, q));
}

void BM_closed_forms(benchmark::State& state) {
    // The chapter's closed forms over 1000 strikes: rn_expected_payoff,
    // digitals and exercise probabilities.
    std::vector<double> strikes(1000);
    for (std::size_t i = 0; i < strikes.size(); ++i) strikes[i] = S0 * (0.5 + static_cast<double>(i) / 1000.0);
    for (auto _ : state) {
        double sum = 0.0;
        for (const double k : strikes) {
            sum += ch04::rn_expected_payoff(OptionType::Call, S0, k, T, sigma, r, q);
            sum += ch04::cash_or_nothing(OptionType::Put, S0, k, T, sigma, r, q);
            sum += ch04::asset_or_nothing(OptionType::Call, S0, k, T, sigma, r, q);
            const auto p = ch04::exercise_probabilities(S0, k, T, sigma, r, q);
            benchmark::DoNotOptimize(p);
        }
        benchmark::DoNotOptimize(sum);
    }
    set_items(state, strikes.size(), "strikes");
}

void BM_monte_carlo(benchmark::State& state, bool antithetic) {
    const auto n = static_cast<std::size_t>(state.range(0));
    const auto call = [](double s) { return std::max(s - K, 0.0); };
    const double exact = p4mfe::ch03::bs_price(OptionType::Call, S0, K, T, sigma, r, q);
    const auto res = ch04::monte_carlo_price(call, S0, T, sigma, r, q, n, 42, antithetic);
    if (!check(state, std::abs(res.price - exact) < 5.0 * res.std_error + 1e-9, "Monte Carlo price off by > 5 SE"))
        return;
    for (auto _ : state) benchmark::DoNotOptimize(ch04::monte_carlo_price(call, S0, T, sigma, r, q, n, 42, antithetic));
    state.counters["std_error"] = res.std_error;
    state.SetComplexityN(static_cast<std::int64_t>(n));
    set_items(state, n, "paths");
}

const bool registered = [] {
    benchmark::RegisterBenchmark("W4/ch04/risk_neutral_price/call", BM_quadrature_call);
    benchmark::RegisterBenchmark("W4/ch04/risk_neutral_price/digital", BM_quadrature_digital);
    benchmark::RegisterBenchmark("W4/ch04/risk_neutral_expectation/power", BM_quadrature_power);
    benchmark::RegisterBenchmark("W1/ch04/closed_forms", BM_closed_forms);
    const auto ns = p4mfe::bench::by_tier<std::vector<std::int64_t>>({10'000}, {10'000, 1'000'000, 10'000'000},
                                                                     {1'000'000, 10'000'000, 100'000'000});
    for (const bool anti : {true, false}) {
        auto* b = benchmark::RegisterBenchmark(std::string("W5/ch04/monte_carlo_price/") + (anti ? "antithetic" : "plain"),
                                               BM_monte_carlo, anti);
        for (const auto n : ns) b->Arg(n);
        b->Complexity(benchmark::oN)->Unit(benchmark::kMillisecond);
    }
    return true;
}();

}  // namespace
