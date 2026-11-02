// Remaining chapter 4 paths (coverage sweep): the lognormal asset model over
// the option chain (moments, quantiles, confidence intervals, in-the-money
// probabilities), density transforms, and sums/products of independent
// variables.
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/normal_rv.hpp>
#include <p4mfe/ch04/asset_model.hpp>
#include <p4mfe/ch04/change_of_variables.hpp>
#include <p4mfe/ch04/in_the_money.hpp>
#include <p4mfe/ch04/independent.hpp>
#include <p4mfe/ch04/lognormal.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

namespace ch03 = p4mfe::ch03;
namespace ch04 = p4mfe::ch04;
using p4mfe::bench::check;
using p4mfe::bench::set_items;

void BM_asset_model(benchmark::State& state) {
    // Real-world drift 8%; one evaluation of every model quantity per option.
    const auto spec = p4mfe::bench::chain_spec_for_tier();
    const auto chain = p4mfe::bench::make_option_chain(spec);
    const double mu = 0.08;
    {
        const auto& o = chain.front();
        const auto ci = ch04::price_confidence_interval(0.95, o.S, o.T, o.sigma, mu, spec.q);
        const double inside = ch04::price_cdf(ci.hi, o.S, o.T, o.sigma, mu, spec.q) -
                              ch04::price_cdf(ci.lo, o.S, o.T, o.sigma, mu, spec.q);
        if (!check(state, std::abs(inside - 0.95) < 1e-9, "confidence interval misses 95%")) return;
    }
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& o : chain) {
            const auto lr = ch04::log_return_distribution(o.T, o.sigma, mu, spec.q);
            sum += lr.mean + ch04::price_mean(o.S, o.T, mu, spec.q);
            sum += ch04::price_variance(o.S, o.T, o.sigma, mu, spec.q);
            sum += ch04::price_median(o.S, o.T, o.sigma, mu, spec.q);
            sum += ch04::price_pdf(o.K, o.S, o.T, o.sigma, mu, spec.q);
            sum += ch04::price_cdf(o.K, o.S, o.T, o.sigma, mu, spec.q);
            sum += ch04::price_quantile(0.25, o.S, o.T, o.sigma, mu, spec.q);
            const auto ci = ch04::price_confidence_interval(0.9, o.S, o.T, o.sigma, mu, spec.q);
            sum += ci.hi - ci.lo;
            sum += ch04::prob_itm(o.type, o.S, o.K, o.T, o.sigma, mu, spec.q);
            sum += ch04::prob_between(0.9 * o.K, 1.1 * o.K, o.S, o.T, o.sigma, mu, spec.q);
        }
        benchmark::DoNotOptimize(sum);
    }
    set_items(state, chain.size(), "options");
}

void BM_density_transforms(benchmark::State& state) {
    // Densities of aX+b, e^X, X^2 and generic transforms on a 10,000-point grid.
    const auto phi = [](double x) { return ch03::normal_pdf(x, 0.0, 1.0); };
    const auto lin = ch04::linear_transform_pdf(phi, 2.0, 1.0);
    const auto ex = ch04::exp_transform_pdf(phi);
    const auto sq = ch04::square_transform_pdf(phi);
    const auto gen = ch04::transformed_pdf(phi, [](double y) { return std::log(y); }, [](double y) { return 1.0 / y; });
    const auto gen_r = ch04::transformed_pdf(
        phi, [](double y) { return std::log(y); }, [](double y) { return 1.0 / y; }, [](double y) { return y > 0.0; });
    const auto num = ch04::transformed_pdf_numeric(phi, [](double y) { return std::log(y); }, [](double y) { return y > 0.0; });
    const std::size_t n = 10'000;
    if (!check(state, std::abs(ex(1.5) - ch03::lognormal_pdf(1.5, 0.0, 1.0)) < 1e-12 && std::abs(num(1.5) - ex(1.5)) < 1e-6,
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
    set_items(state, n, "points");
}

void BM_independent(benchmark::State& state) {
    // 500 independent normals; one convolution density point; lognormal helpers.
    const std::size_t n = 500;
    std::vector<double> a(n);
    std::vector<ch03::NormalParams> X(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = 1.0 / static_cast<double>(n);
        X[i] = {0.0005 * static_cast<double>(i % 7), 0.0001 * static_cast<double>(1 + i % 5)};
    }
    const auto phi = [](double x) { return ch03::normal_pdf(x, 0.0, 1.0); };
    const auto joint = ch04::independent_joint_pdf(phi, phi);
    // N(0,1) + N(0,1) = N(0,2).
    const double conv = ch04::sum_pdf(phi, phi, 0.5);
    if (!check(state, std::abs(conv - ch03::normal_pdf(0.5, 0.0, std::sqrt(2.0))) < 1e-8, "sum_pdf is off")) return;
    for (auto _ : state) {
        const auto c = ch04::independent_normal_combination(a, X);
        const auto s = ch04::independent_normal_sum(X);
        const auto p = ch04::independent_lognormal_product(a, X);
        const auto p1 = ch04::independent_lognormal_product(X);
        double sum = c.mean + s.variance + p.mean + p1.variance + joint(0.1, -0.2);
        sum += ch04::sum_pdf(phi, phi, 0.5);
        sum += ch04::lognormal_moment(3.0, 0.05, 0.2) + ch04::lognormal_quantile(0.9, 0.05, 0.2);
        sum += ch04::lognormal_median(0.05) + ch04::lognormal_mode(0.05, 0.2);
        sum += ch04::lognormal_params_from_moments(1.05, 0.04).variance;
        benchmark::DoNotOptimize(sum);
    }
    set_items(state, n, "variables");
}

const bool registered = [] {
    benchmark::RegisterBenchmark("W2/ch04/asset_model", BM_asset_model);
    benchmark::RegisterBenchmark("W2/ch04/density_transforms", BM_density_transforms);
    benchmark::RegisterBenchmark("W2/ch04/independent+lognormal", BM_independent)->Unit(benchmark::kMicrosecond);
    return true;
}();

}  // namespace
