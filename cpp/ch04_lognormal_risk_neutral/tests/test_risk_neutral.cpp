// Sections 4.4-4.7: lognormal price model, risk-neutral pricing, probability
// of finishing in the money, and N(d1), N(d2).

#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch04/asset_model.hpp>
#include <p4mfe/ch04/in_the_money.hpp>
#include <p4mfe/ch04/lognormal.hpp>
#include <p4mfe/ch04/risk_neutral.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

using namespace p4mfe::ch04;
namespace ch02 = p4mfe::ch02;
namespace ch03 = p4mfe::ch03;
using p4mfe::norm_cdf;
using p4mfe::norm_pdf;

namespace {
constexpr double S0 = 40, K = 42, T = 0.75, sigma = 0.3, r = 0.04, q = 0.01, mu = 0.10;
}

// ---------------------------------------------------------------- 4.4 ------
TEST(Ch04AssetModel, MomentsAndMedian) {
    EXPECT_NEAR(price_mean(S0, T, mu, q), S0 * std::exp((mu - q) * T), 1e-13);
    const auto L = log_price_distribution(S0, T, sigma, mu, q);
    EXPECT_NEAR(price_mean(S0, T, mu, q), lognormal_mean(L.mean, L.stddev()), 1e-12);
    EXPECT_NEAR(price_variance(S0, T, sigma, mu, q), lognormal_variance(L.mean, L.stddev()), 1e-11);
    EXPECT_NEAR(price_cdf(price_median(S0, T, sigma, mu, q), S0, T, sigma, mu, q), 0.5, 1e-15);
    EXPECT_LT(price_median(S0, T, sigma, mu, q), price_mean(S0, T, mu, q));
    const auto R = log_return_distribution(T, sigma, mu, q);
    EXPECT_NEAR(R.mean, L.mean - std::log(S0), 1e-15);
    EXPECT_DOUBLE_EQ(R.variance, L.variance);
}

TEST(Ch04AssetModel, DensityIntegratesToMoments) {
    const ch03::ContinuousDistribution S{[](double x) { return price_pdf(x, S0, T, sigma, mu, q); }, 0.0,
                                         ch02::infinity};
    EXPECT_NEAR(ch03::mean(S), price_mean(S0, T, mu, q), 1e-7);
    EXPECT_NEAR(ch03::cdf(S, 45.0), price_cdf(45.0, S0, T, sigma, mu, q), 1e-8);
}

TEST(Ch04AssetModel, ConfidenceInterval) {
    // The book's style of question: a 95% range for S_T.
    const auto [lo, hi] = price_confidence_interval(0.95, S0, T, sigma, mu, q);
    EXPECT_NEAR(price_cdf(hi, S0, T, sigma, mu, q) - price_cdf(lo, S0, T, sigma, mu, q), 0.95, 1e-13);
    EXPECT_NEAR(price_cdf(lo, S0, T, sigma, mu, q), 0.025, 1e-14);
    const double m = price_median(S0, T, sigma, mu, q);
    EXPECT_NEAR(lo * hi, m * m, 1e-9);           // symmetric in log space
    EXPECT_GT(hi - m, m - lo);                   // skewed right in price space
    const double z = 1.959963984540054;
    EXPECT_NEAR(lo, m * std::exp(-z * sigma * std::sqrt(T)), 1e-11);
    EXPECT_NEAR(price_quantile(0.975, S0, T, sigma, mu, q), hi, 1e-11);
}

// ---------------------------------------------------------------- 4.5 ------
TEST(Ch04RiskNeutral, ClosedFormDiscountedIsBlackScholes) {
    for (auto type : {OptionType::Call, OptionType::Put})
        EXPECT_NEAR(std::exp(-r * T) * rn_expected_payoff(type, S0, K, T, sigma, r, q),
                    ch03::bs_price(type, S0, K, T, sigma, r, q), 1e-13);
}

TEST(Ch04RiskNeutral, IntegralOfPayoffIsBlackScholes) {
    const auto call = [](double s) { return std::max(s - K, 0.0); };
    const auto put = [](double s) { return std::max(K - s, 0.0); };
    EXPECT_NEAR(risk_neutral_price(call, S0, T, sigma, r, q), ch03::bs_call(S0, K, T, sigma, r, q), 1e-7);
    EXPECT_NEAR(risk_neutral_price(put, S0, T, sigma, r, q), ch03::bs_put(S0, K, T, sigma, r, q), 1e-7);
    // Under the risk-neutral measure the forward is the expected price.
    EXPECT_NEAR(risk_neutral_expectation([](double s) { return s; }, S0, T, sigma, r, q),
                S0 * std::exp((r - q) * T), 1e-7);
}

TEST(Ch04RiskNeutral, DerivationRestrictsToExerciseRegion) {
    // The derivation: S_T > K iff z > -d2; on that region the integrand is smooth.
    const auto [d1, d2] = ch03::bs_d1_d2(S0, K, T, sigma, r, q);
    const auto ST = [](double z) { return price_from_normal(z, S0, T, sigma, r, q); };
    EXPECT_NEAR(ST(-d2), K, 1e-12);
    // phi(z) underflows to 0 long before S_T(z) overflows; skip S_T there.
    const auto weighted = [&](double z) {
        const double p = norm_pdf(z);
        return p == 0.0 ? 0.0 : ST(z) * p;
    };
    const double asset_part = ch02::improper_integral(weighted, -d2, ch02::infinity, 1e-10).value;
    const double cash_part = ch02::improper_integral([](double z) { return norm_pdf(z); }, -d2, ch02::infinity, 1e-10).value;
    EXPECT_NEAR(asset_part, S0 * std::exp((r - q) * T) * norm_cdf(d1), 1e-8);  // completing the square
    EXPECT_NEAR(cash_part, norm_cdf(d2), 1e-9);
}

TEST(Ch04RiskNeutral, MonteCarloWithinStandardErrors) {
    const auto call = [](double s) { return std::max(s - K, 0.0); };
    const double bs = ch03::bs_call(S0, K, T, sigma, r, q);
    for (bool anti : {false, true}) {
        const auto mc = monte_carlo_price(call, S0, T, sigma, r, q, 200'000, 7, anti);
        EXPECT_GT(mc.std_error, 0.0);
        EXPECT_LT(mc.std_error, 0.03);
        EXPECT_NEAR(mc.price, bs, 4.0 * mc.std_error) << anti;
    }
    // Under the real-world drift the discounted expectation is not the price.
    const double wrong = std::exp(-r * T) * risk_neutral_expectation(call, S0, T, sigma, mu, q);
    EXPECT_GT(wrong, bs + 0.1);
}

// ------------------------------------------------------------ 4.6, 4.7 -----
TEST(Ch04InTheMoney, ProbabilitiesUnderBothMeasures) {
    const auto [d1, d2] = ch03::bs_d1_d2(S0, K, T, sigma, r, q);
    EXPECT_NEAR(prob_itm(OptionType::Call, S0, K, T, sigma, r, q), norm_cdf(d2), 1e-15);
    EXPECT_NEAR(prob_itm(OptionType::Put, S0, K, T, sigma, r, q), norm_cdf(-d2), 1e-15);
    // Real-world: replace r by mu in d2.
    const double d2_mu = d2 + (mu - r) * std::sqrt(T) / sigma;
    EXPECT_NEAR(prob_itm(OptionType::Call, S0, K, T, sigma, mu, q), norm_cdf(d2_mu), 1e-14);
    EXPECT_GT(prob_itm(OptionType::Call, S0, K, T, sigma, mu, q), prob_itm(OptionType::Call, S0, K, T, sigma, r, q));
    EXPECT_NEAR(prob_between(35, 45, S0, T, sigma, mu, q),
                price_cdf(45, S0, T, sigma, mu, q) - price_cdf(35, S0, T, sigma, mu, q), 1e-15);
}

TEST(Ch04InTheMoney, CallIsAssetMinusCashDigital) {
    for (auto type : {OptionType::Call, OptionType::Put}) {
        const double aon = asset_or_nothing(type, S0, K, T, sigma, r, q);
        const double con = cash_or_nothing(type, S0, K, T, sigma, r, q);
        const double sign = type == OptionType::Call ? 1.0 : -1.0;
        EXPECT_NEAR(sign * (aon - K * con), ch03::bs_price(type, S0, K, T, sigma, r, q), 1e-13);
    }
    // Digital call + digital put = a zero coupon bond; AoN call + AoN put = the asset.
    EXPECT_NEAR(cash_or_nothing(OptionType::Call, S0, K, T, sigma, r, q) +
                    cash_or_nothing(OptionType::Put, S0, K, T, sigma, r, q),
                std::exp(-r * T), 1e-15);
    EXPECT_NEAR(asset_or_nothing(OptionType::Call, S0, K, T, sigma, r, q) +
                    asset_or_nothing(OptionType::Put, S0, K, T, sigma, r, q),
                S0 * std::exp(-q * T), 1e-13);
}

TEST(Ch04InTheMoney, FinancialMeaningOfNd1Nd2) {
    const auto [n_d1, n_d2] = exercise_probabilities(S0, K, T, sigma, r, q);
    EXPECT_GT(n_d1, n_d2);
    // N(d2): risk-neutral exercise probability, the price of the digital grown at r.
    EXPECT_NEAR(n_d2, std::exp(r * T) * cash_or_nothing(OptionType::Call, S0, K, T, sigma, r, q), 1e-15);
    // N(d1) = E_RN[S_T 1{S_T > K}] / E_RN[S_T].
    const double fwd = S0 * std::exp((r - q) * T);
    EXPECT_NEAR(n_d1, std::exp(r * T) * asset_or_nothing(OptionType::Call, S0, K, T, sigma, r, q) / fwd, 1e-14);
    // e^(-qT) N(d1) is Delta; -e^(-rT) N(d2) is dC/dK.
    const auto g = ch03::bs_greeks(OptionType::Call, S0, K, T, sigma, r, q);
    EXPECT_NEAR(g.delta, std::exp(-q * T) * n_d1, 1e-15);
    const double dC_dK = p4mfe::ch01::derivative([](double k) { return ch03::bs_call(S0, k, T, sigma, r, q); }, K);
    EXPECT_NEAR(dC_dK, -std::exp(-r * T) * n_d2, 1e-8);
}
