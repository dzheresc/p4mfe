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
  EXPECT_NEAR(Ex09ProbItm(), 0.271525, 1e-6);
  EXPECT_NEAR(Ex09RnProbItm(), 0.304331, 1e-6);
  EXPECT_NEAR(Ex09NMinusD2(), Ex09RnProbItm(), 1e-12);
  // Independent check: P(S_T < K) by integrating the density.
  const Ex09Market m;
  const double prob =
      RiskNeutralExpectation([&](double x) { return x < m.strike ? 1.0 : 0.0; },
                             m.spot, m.maturity, m.sigma, m.mu, 0.0, 1e-10);
  EXPECT_NEAR(prob, Ex09ProbItm(), 1e-6);
}

TEST(Ch04Ex10, AtmCallDelta) {
  for (double r : {0.0, 0.02, 0.08})
    for (double sigma : {0.1, 0.3, 0.6})
      for (double maturity : {0.05, 1.0, 10.0}) {
        // (i) q = 0: Delta > 1/2, and agrees with Black-Scholes.
        const double d = Ex10AtmCallDelta(r, 0.0, sigma, maturity);
        EXPECT_GT(d, 0.5);
        EXPECT_NEAR(
            d,
            ch03::BsGreeks(OptionType::kCall, 100, 100, maturity, sigma, r)
                .delta,
            1e-12);
        // (ii) threshold q*.
        const double qs = Ex10ThresholdQ(r, sigma, maturity);
        EXPECT_NEAR(Ex10AtmCallDelta(r, qs, sigma, maturity), 0.5, 1e-12);
        EXPECT_GT(Ex10AtmCallDelta(r, 0.9 * qs, sigma, maturity), 0.5);
        EXPECT_LT(Ex10AtmCallDelta(r, 1.1 * qs + 1e-6, sigma, maturity), 0.5);
        EXPECT_LT(qs, r + 0.5 * sigma * sigma);
      }
  EXPECT_NEAR(Ex10ThresholdQ(0.05, 0.2, 0.25), 0.062149, 1e-6);
  EXPECT_NEAR(Ex10ThresholdQ(0.05, 0.2, 1.0), 0.055644, 1e-6);
  EXPECT_NEAR(Ex10ThresholdQ(0.05, 0.2, 5.0), 0.042822, 1e-6);
  // The linearised threshold is close for short maturities.
  EXPECT_NEAR(Ex10ThresholdQApprox(0.05, 0.2, 0.25),
              Ex10ThresholdQ(0.05, 0.2, 0.25), 1e-4);
}

TEST(Ch04Ex11, PutByIntegration) {
  const double put = Ex11PutBs();
  EXPECT_NEAR(put, 3.918999, 1e-6);
  for (auto rule : {ch02::Rule::kMidpoint, ch02::Rule::kSimpson}) {
    const auto steps = Ex11Run(rule);
    ASSERT_GE(steps.size(), 2u);
    EXPECT_NEAR(steps.back().value, put, 5e-7);  // six decimal digits
    // Errors shrink with every doubling once n is moderate.
    for (std::size_t i = 3; i < steps.size(); ++i)
      EXPECT_LT(std::abs(steps[i].error), std::abs(steps[i - 1].error));
  }
  EXPECT_EQ(Ex11Run(ch02::Rule::kMidpoint).back().n, 8192u);
  EXPECT_EQ(Ex11Run(ch02::Rule::kSimpson).back().n, 128u);
  // Observed orders 2 and 4.
  const auto f = [](double x) { return Ex11Integrand(x); };
  EXPECT_NEAR(ch02::ObservedOrder(f, 0.0, 50.0, put, 64, ch02::Rule::kMidpoint),
              2.0, 0.1);
  EXPECT_NEAR(ch02::ObservedOrder(f, 0.0, 50.0, put, 32, ch02::Rule::kSimpson),
              4.0, 0.2);
}

namespace {
struct Case {
  double spot, strike, maturity, sigma, r, q;
};
const Case kCases[] = {{50, 45, 0.5, 0.2, 0.06, 0.0},
                       {100, 110, 1.5, 0.35, 0.03, 0.0},
                       {40, 38, 0.25, 0.15, 0.01, 0.04},
                       {75, 75, 2.0, 0.25, 0.05, 0.02}};

// e^(-rT) Integral_{z_lo}^{12} g(S_T(z)) phi(z) dz with Simpson's rule. The
// payoffs below are smooth for z > z_lo (z_lo = where the option is struck),
// so the kink sits at an endpoint and Simpson converges at order 4.
template <class G>
double RnPriceAbove(G g, double z_lo, const Case& c) {
  const auto f = [&](double z) {
    return g(PriceFromNormal(z, c.spot, c.maturity, c.sigma, c.r, c.q)) *
           NormPdf(z);
  };
  return std::exp(-c.r * c.maturity) * ch02::SimpsonRule(f, z_lo, 12.0, 20000);
}
double ZAt(double strike, const Case& c) {
  return -ch03::BsD1D2(c.spot, strike, c.maturity, c.sigma, c.r, c.q).d2;
}
}  // namespace

TEST(Ch04Ex12, SquaredCall) {
  for (const auto& c : kCases) {
    const double strike = c.strike;
    const double numeric =
        RnPriceAbove([strike](double x) { return (x - strike) * (x - strike); },
                     ZAt(strike, c), c);
    EXPECT_NEAR(
        Ex12SquaredCall(c.spot, c.strike, c.maturity, c.sigma, c.r, c.q),
        numeric, 1e-6 * std::max(1.0, numeric));
  }
}

TEST(Ch04Ex13, PowerCall) {
  for (const auto& c : kCases) {
    // alpha = 1 is the Black-Scholes call.
    EXPECT_NEAR(
        Ex13PowerCall(c.spot, c.strike, c.maturity, c.sigma, c.r, 1.0, c.q),
        ch03::BsCall(c.spot, c.strike, c.maturity, c.sigma, c.r, c.q), 1e-10);
    for (double alpha : {0.5, 1.5, 2.0}) {
      const double strike = std::pow(c.strike, alpha);
      const double numeric = RnPriceAbove(
          [strike, alpha](double x) { return std::pow(x, alpha) - strike; },
          ZAt(c.strike, c), c);
      EXPECT_NEAR(
          Ex13PowerCall(c.spot, strike, c.maturity, c.sigma, c.r, alpha, c.q),
          numeric, 1e-6 * std::max(1.0, numeric));
    }
  }
}

TEST(Ch04Ex14, NormalModel) {
  const double spot0 = 50, strike = 52, maturity = 0.75, sigma = 8.0, r = 0.04;
  const double forward = Ex14BookForward(spot0, r, maturity);
  // (i) closed form vs direct integration against phi.
  const ch03::ContinuousDistribution std_normal{
      [](double z) { return NormPdf(z); }, -ch02::kInfinity, ch02::kInfinity};
  const double s = sigma * std::sqrt(maturity);
  const double numeric =
      std::exp(-r * maturity) *
      ch03::ExpectedValue(
          std_normal,
          [&](double z) { return std::max(forward + s * z - strike, 0.0); },
          1e-11);
  EXPECT_NEAR(Ex14Call(spot0, strike, maturity, sigma, r), numeric, 1e-7);
  // (ii) parity.
  EXPECT_NEAR(Ex14Call(spot0, strike, maturity, sigma, r) -
                  Ex14PutParity(spot0, strike, maturity, sigma, r),
              spot0 - strike * std::exp(-r * maturity), 1e-12);
  // The direct put differs from the parity put by e^(-rT) F - S0 ...
  EXPECT_NEAR(Ex14PutParity(spot0, strike, maturity, sigma, r) -
                  Ex14PutForward(forward, strike, maturity, sigma, r),
              std::exp(-r * maturity) * forward - spot0, 1e-12);
  // ... and the two agree with the martingale forward S0 e^(rT).
  const double forward_martingale = spot0 * std::exp(r * maturity);
  EXPECT_NEAR(
      Ex14CallForward(forward_martingale, strike, maturity, sigma, r) -
          Ex14PutForward(forward_martingale, strike, maturity, sigma, r),
      spot0 - strike * std::exp(-r * maturity), 1e-12);
}

TEST(Ch04Ex15, InversePayoffs) {
  for (const auto& c : kCases) {
    const double v1 = RiskNeutralPrice([](double x) { return 1.0 / x; }, c.spot,
                                       c.maturity, c.sigma, c.r, c.q, 1e-12);
    EXPECT_NEAR(Ex15InverseClaim(c.spot, c.maturity, c.sigma, c.r, c.q), v1,
                1e-9);
    const double strike = c.strike;
    const double v2 =
        RnPriceAbove([strike](double x) { return 1.0 / strike - 1.0 / x; },
                     ZAt(strike, c), c);
    EXPECT_NEAR(
        Ex15InverseCall(c.spot, c.strike, c.maturity, c.sigma, c.r, c.q), v2,
        1e-9);
  }
}
