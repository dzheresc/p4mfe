#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <p4mfe/ch04/exercises/ex09_10_itm_delta.hpp>
#include <p4mfe/ch04/exercises/ex11_put_integration.hpp>
#include <p4mfe/ch04/exercises/ex12_13_power_payoffs.hpp>
#include <p4mfe/ch04/exercises/ex14_15_normal_model_inverse.hpp>
#include <p4mfe/ch04/risk_neutral.hpp>

using namespace p4mfe;
using namespace p4mfe::ch04;
using namespace p4mfe::ch04::ex;

TEST(Ch04Ex09, PutInTheMoney) {
  EXPECT_NEAR(ex09_prob_itm(), 0.271525, 1e-6);
  EXPECT_NEAR(ex09_rn_prob_itm(), 0.304331, 1e-6);
  EXPECT_NEAR(ex09_n_minus_d2(), ex09_rn_prob_itm(), 1e-12);
  // Independent check: P(S_T < K) by integrating the density.
  const Ex09Market m;
  const double P =
      risk_neutral_expectation([&](double x) { return x < m.K ? 1.0 : 0.0; },
                               m.S, m.T, m.sigma, m.mu, 0.0, 1e-10);
  EXPECT_NEAR(P, ex09_prob_itm(), 1e-6);
}

TEST(Ch04Ex10, AtmCallDelta) {
  for (double r : {0.0, 0.02, 0.08})
    for (double sigma : {0.1, 0.3, 0.6})
      for (double T : {0.05, 1.0, 10.0}) {
        // (i) q = 0: Delta > 1/2, and agrees with Black-Scholes.
        const double d = ex10_atm_call_delta(r, 0.0, sigma, T);
        EXPECT_GT(d, 0.5);
        EXPECT_NEAR(
            d, ch03::bs_greeks(OptionType::Call, 100, 100, T, sigma, r).delta,
            1e-12);
        // (ii) threshold q*.
        const double qs = ex10_threshold_q(r, sigma, T);
        EXPECT_NEAR(ex10_atm_call_delta(r, qs, sigma, T), 0.5, 1e-12);
        EXPECT_GT(ex10_atm_call_delta(r, 0.9 * qs, sigma, T), 0.5);
        EXPECT_LT(ex10_atm_call_delta(r, 1.1 * qs + 1e-6, sigma, T), 0.5);
        EXPECT_LT(qs, r + 0.5 * sigma * sigma);
      }
  EXPECT_NEAR(ex10_threshold_q(0.05, 0.2, 0.25), 0.062149, 1e-6);
  EXPECT_NEAR(ex10_threshold_q(0.05, 0.2, 1.0), 0.055644, 1e-6);
  EXPECT_NEAR(ex10_threshold_q(0.05, 0.2, 5.0), 0.042822, 1e-6);
  // The linearised threshold is close for short maturities.
  EXPECT_NEAR(ex10_threshold_q_approx(0.05, 0.2, 0.25),
              ex10_threshold_q(0.05, 0.2, 0.25), 1e-4);
}

TEST(Ch04Ex11, PutByIntegration) {
  const double P = ex11_put_bs();
  EXPECT_NEAR(P, 3.918999, 1e-6);
  for (auto rule : {ch02::Rule::Midpoint, ch02::Rule::Simpson}) {
    const auto steps = ex11_run(rule);
    ASSERT_GE(steps.size(), 2u);
    EXPECT_NEAR(steps.back().value, P, 5e-7);  // six decimal digits
    // Errors shrink with every doubling once n is moderate.
    for (std::size_t i = 3; i < steps.size(); ++i)
      EXPECT_LT(std::abs(steps[i].error), std::abs(steps[i - 1].error));
  }
  EXPECT_EQ(ex11_run(ch02::Rule::Midpoint).back().n, 8192u);
  EXPECT_EQ(ex11_run(ch02::Rule::Simpson).back().n, 128u);
  // Observed orders 2 and 4.
  const auto f = [](double x) { return ex11_integrand(x); };
  EXPECT_NEAR(ch02::observed_order(f, 0.0, 50.0, P, 64, ch02::Rule::Midpoint),
              2.0, 0.1);
  EXPECT_NEAR(ch02::observed_order(f, 0.0, 50.0, P, 32, ch02::Rule::Simpson),
              4.0, 0.2);
}

namespace {
struct Case {
  double S, K, T, sigma, r, q;
};
const Case cases[] = {{50, 45, 0.5, 0.2, 0.06, 0.0},
                      {100, 110, 1.5, 0.35, 0.03, 0.0},
                      {40, 38, 0.25, 0.15, 0.01, 0.04},
                      {75, 75, 2.0, 0.25, 0.05, 0.02}};

// e^(-rT) Integral_{z_lo}^{12} g(S_T(z)) phi(z) dz with Simpson's rule. The
// payoffs below are smooth for z > z_lo (z_lo = where the option is struck),
// so the kink sits at an endpoint and Simpson converges at order 4.
template <class G>
double rn_price_above(G g, double z_lo, const Case& c) {
  const auto f = [&](double z) {
    return g(price_from_normal(z, c.S, c.T, c.sigma, c.r, c.q)) * norm_pdf(z);
  };
  return std::exp(-c.r * c.T) * ch02::simpson_rule(f, z_lo, 12.0, 20000);
}
double z_at(double K, const Case& c) {
  return -ch03::bs_d1_d2(c.S, K, c.T, c.sigma, c.r, c.q).d2;
}
}  // namespace

TEST(Ch04Ex12, SquaredCall) {
  for (const auto& c : cases) {
    const double K = c.K;
    const double numeric = rn_price_above(
        [K](double x) { return (x - K) * (x - K); }, z_at(K, c), c);
    EXPECT_NEAR(ex12_squared_call(c.S, c.K, c.T, c.sigma, c.r, c.q), numeric,
                1e-6 * std::max(1.0, numeric));
  }
}

TEST(Ch04Ex13, PowerCall) {
  for (const auto& c : cases) {
    // alpha = 1 is the Black-Scholes call.
    EXPECT_NEAR(ex13_power_call(c.S, c.K, c.T, c.sigma, c.r, 1.0, c.q),
                ch03::bs_call(c.S, c.K, c.T, c.sigma, c.r, c.q), 1e-10);
    for (double alpha : {0.5, 1.5, 2.0}) {
      const double K = std::pow(c.K, alpha);
      const double numeric = rn_price_above(
          [K, alpha](double x) { return std::pow(x, alpha) - K; }, z_at(c.K, c),
          c);
      EXPECT_NEAR(ex13_power_call(c.S, K, c.T, c.sigma, c.r, alpha, c.q),
                  numeric, 1e-6 * std::max(1.0, numeric));
    }
  }
}

TEST(Ch04Ex14, NormalModel) {
  const double S0 = 50, K = 52, T = 0.75, sigma = 8.0, r = 0.04;
  const double F = ex14_book_forward(S0, r, T);
  // (i) closed form vs direct integration against phi.
  const ch03::ContinuousDistribution Z{[](double z) { return norm_pdf(z); },
                                       -ch02::infinity, ch02::infinity};
  const double s = sigma * std::sqrt(T);
  const double numeric =
      std::exp(-r * T) *
      ch03::expected_value(
          Z, [&](double z) { return std::max(F + s * z - K, 0.0); }, 1e-11);
  EXPECT_NEAR(ex14_call(S0, K, T, sigma, r), numeric, 1e-7);
  // (ii) parity.
  EXPECT_NEAR(
      ex14_call(S0, K, T, sigma, r) - ex14_put_parity(S0, K, T, sigma, r),
      S0 - K * std::exp(-r * T), 1e-12);
  // The direct put differs from the parity put by e^(-rT) F - S0 ...
  EXPECT_NEAR(
      ex14_put_parity(S0, K, T, sigma, r) - ex14_put_forward(F, K, T, sigma, r),
      std::exp(-r * T) * F - S0, 1e-12);
  // ... and the two agree with the martingale forward S0 e^(rT).
  const double Fm = S0 * std::exp(r * T);
  EXPECT_NEAR(ex14_call_forward(Fm, K, T, sigma, r) -
                  ex14_put_forward(Fm, K, T, sigma, r),
              S0 - K * std::exp(-r * T), 1e-12);
}

TEST(Ch04Ex15, InversePayoffs) {
  for (const auto& c : cases) {
    const double v1 = risk_neutral_price([](double x) { return 1.0 / x; }, c.S,
                                         c.T, c.sigma, c.r, c.q, 1e-12);
    EXPECT_NEAR(ex15_inverse_claim(c.S, c.T, c.sigma, c.r, c.q), v1, 1e-9);
    const double K = c.K;
    const double v2 = rn_price_above(
        [K](double x) { return 1.0 / K - 1.0 / x; }, z_at(K, c), c);
    EXPECT_NEAR(ex15_inverse_call(c.S, c.K, c.T, c.sigma, c.r, c.q), v2, 1e-9);
  }
}
