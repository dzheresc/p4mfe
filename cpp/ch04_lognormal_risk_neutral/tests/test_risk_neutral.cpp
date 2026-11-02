// Sections 4.4-4.7: lognormal price model, risk-neutral pricing, probability
// of finishing in the money, and N(d1), N(d2).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch04/asset_model.hpp>
#include <p4mfe/ch04/in_the_money.hpp>
#include <p4mfe/ch04/lognormal.hpp>
#include <p4mfe/ch04/risk_neutral.hpp>

using namespace p4mfe::ch04;
namespace ch02 = p4mfe::ch02;
namespace ch03 = p4mfe::ch03;
using p4mfe::norm_cdf;
using p4mfe::norm_pdf;

namespace {
constexpr double kS0 = 40, kK = 42, kT = 0.75, kSigma = 0.3, kR = 0.04,
                 kQ = 0.01, kMu = 0.10;
}

// ---------------------------------------------------------------- 4.4 ------
TEST(Ch04AssetModel, MomentsAndMedian) {
  EXPECT_NEAR(PriceMean(kS0, kT, kMu, kQ), kS0 * std::exp((kMu - kQ) * kT),
              1e-13);
  const auto log_price = LogPriceDistribution(kS0, kT, kSigma, kMu, kQ);
  EXPECT_NEAR(PriceMean(kS0, kT, kMu, kQ),
              LognormalMean(log_price.mean, log_price.Stddev()), 1e-12);
  EXPECT_NEAR(PriceVariance(kS0, kT, kSigma, kMu, kQ),
              LognormalVariance(log_price.mean, log_price.Stddev()), 1e-11);
  EXPECT_NEAR(
      PriceCdf(PriceMedian(kS0, kT, kSigma, kMu, kQ), kS0, kT, kSigma, kMu, kQ),
      0.5, 1e-15);
  EXPECT_LT(PriceMedian(kS0, kT, kSigma, kMu, kQ), PriceMean(kS0, kT, kMu, kQ));
  const auto log_return = LogReturnDistribution(kT, kSigma, kMu, kQ);
  EXPECT_NEAR(log_return.mean, log_price.mean - std::log(kS0), 1e-15);
  EXPECT_DOUBLE_EQ(log_return.variance, log_price.variance);
}

TEST(Ch04AssetModel, DensityIntegratesToMoments) {
  const ch03::ContinuousDistribution spot{
      [](double x) { return PricePdf(x, kS0, kT, kSigma, kMu, kQ); }, 0.0,
      ch02::kInfinity};
  EXPECT_NEAR(ch03::Mean(spot), PriceMean(kS0, kT, kMu, kQ), 1e-7);
  EXPECT_NEAR(ch03::Cdf(spot, 45.0), PriceCdf(45.0, kS0, kT, kSigma, kMu, kQ),
              1e-8);
}

TEST(Ch04AssetModel, ConfidenceInterval) {
  // The book's style of question: a 95% range for S_T.
  const auto [lo, hi] = PriceConfidenceInterval(0.95, kS0, kT, kSigma, kMu, kQ);
  EXPECT_NEAR(PriceCdf(hi, kS0, kT, kSigma, kMu, kQ) -
                  PriceCdf(lo, kS0, kT, kSigma, kMu, kQ),
              0.95, 1e-13);
  EXPECT_NEAR(PriceCdf(lo, kS0, kT, kSigma, kMu, kQ), 0.025, 1e-14);
  const double m = PriceMedian(kS0, kT, kSigma, kMu, kQ);
  EXPECT_NEAR(lo * hi, m * m, 1e-9);  // symmetric in log space
  EXPECT_GT(hi - m, m - lo);          // skewed right in price space
  const double z = 1.959963984540054;
  EXPECT_NEAR(lo, m * std::exp(-z * kSigma * std::sqrt(kT)), 1e-11);
  EXPECT_NEAR(PriceQuantile(0.975, kS0, kT, kSigma, kMu, kQ), hi, 1e-11);
}

// ---------------------------------------------------------------- 4.5 ------
TEST(Ch04RiskNeutral, ClosedFormDiscountedIsBlackScholes) {
  for (auto type : {OptionType::kCall, OptionType::kPut})
    EXPECT_NEAR(std::exp(-kR * kT) *
                    RnExpectedPayoff(type, kS0, kK, kT, kSigma, kR, kQ),
                ch03::BsPrice(type, kS0, kK, kT, kSigma, kR, kQ), 1e-13);
}

TEST(Ch04RiskNeutral, IntegralOfPayoffIsBlackScholes) {
  const auto call = [](double s) { return std::max(s - kK, 0.0); };
  const auto put = [](double s) { return std::max(kK - s, 0.0); };
  EXPECT_NEAR(RiskNeutralPrice(call, kS0, kT, kSigma, kR, kQ),
              ch03::BsCall(kS0, kK, kT, kSigma, kR, kQ), 1e-7);
  EXPECT_NEAR(RiskNeutralPrice(put, kS0, kT, kSigma, kR, kQ),
              ch03::BsPut(kS0, kK, kT, kSigma, kR, kQ), 1e-7);
  // Under the risk-neutral measure the forward is the expected price.
  EXPECT_NEAR(RiskNeutralExpectation([](double s) { return s; }, kS0, kT,
                                     kSigma, kR, kQ),
              kS0 * std::exp((kR - kQ) * kT), 1e-7);
}

TEST(Ch04RiskNeutral, DerivationRestrictsToExerciseRegion) {
  // The derivation: S_T > K iff z > -d2; on that region the integrand is
  // smooth.
  const auto [d1, d2] = ch03::BsD1D2(kS0, kK, kT, kSigma, kR, kQ);
  const auto spot_t = [](double z) {
    return PriceFromNormal(z, kS0, kT, kSigma, kR, kQ);
  };
  EXPECT_NEAR(spot_t(-d2), kK, 1e-12);
  // phi(z) underflows to 0 long before S_T(z) overflows; skip S_T there.
  const auto weighted = [&](double z) {
    const double p = norm_pdf(z);
    return p == 0.0 ? 0.0 : spot_t(z) * p;
  };
  const double asset_part =
      ch02::ImproperIntegral(weighted, -d2, ch02::kInfinity, 1e-10).value;
  const double cash_part =
      ch02::ImproperIntegral([](double z) { return norm_pdf(z); }, -d2,
                             ch02::kInfinity, 1e-10)
          .value;
  EXPECT_NEAR(asset_part, kS0 * std::exp((kR - kQ) * kT) * norm_cdf(d1),
              1e-8);  // completing the square
  EXPECT_NEAR(cash_part, norm_cdf(d2), 1e-9);
}

TEST(Ch04RiskNeutral, MonteCarloWithinStandardErrors) {
  const auto call = [](double s) { return std::max(s - kK, 0.0); };
  const double bs = ch03::BsCall(kS0, kK, kT, kSigma, kR, kQ);
  for (bool anti : {false, true}) {
    const auto mc =
        MonteCarloPrice(call, kS0, kT, kSigma, kR, kQ, 200'000, 7, anti);
    EXPECT_GT(mc.std_error, 0.0);
    EXPECT_LT(mc.std_error, 0.03);
    EXPECT_NEAR(mc.price, bs, 4.0 * mc.std_error) << anti;
  }
  // Under the real-world drift the discounted expectation is not the price.
  const double wrong = std::exp(-kR * kT) *
                       RiskNeutralExpectation(call, kS0, kT, kSigma, kMu, kQ);
  EXPECT_GT(wrong, bs + 0.1);
}

// ------------------------------------------------------------ 4.6, 4.7 -----
TEST(Ch04InTheMoney, ProbabilitiesUnderBothMeasures) {
  const auto [d1, d2] = ch03::BsD1D2(kS0, kK, kT, kSigma, kR, kQ);
  EXPECT_NEAR(ProbItm(OptionType::kCall, kS0, kK, kT, kSigma, kR, kQ),
              norm_cdf(d2), 1e-15);
  EXPECT_NEAR(ProbItm(OptionType::kPut, kS0, kK, kT, kSigma, kR, kQ),
              norm_cdf(-d2), 1e-15);
  // Real-world: replace r by mu in d2.
  const double d2_mu = d2 + (kMu - kR) * std::sqrt(kT) / kSigma;
  EXPECT_NEAR(ProbItm(OptionType::kCall, kS0, kK, kT, kSigma, kMu, kQ),
              norm_cdf(d2_mu), 1e-14);
  EXPECT_GT(ProbItm(OptionType::kCall, kS0, kK, kT, kSigma, kMu, kQ),
            ProbItm(OptionType::kCall, kS0, kK, kT, kSigma, kR, kQ));
  EXPECT_NEAR(ProbBetween(35, 45, kS0, kT, kSigma, kMu, kQ),
              PriceCdf(45, kS0, kT, kSigma, kMu, kQ) -
                  PriceCdf(35, kS0, kT, kSigma, kMu, kQ),
              1e-15);
}

TEST(Ch04InTheMoney, CallIsAssetMinusCashDigital) {
  for (auto type : {OptionType::kCall, OptionType::kPut}) {
    const double aon = AssetOrNothing(type, kS0, kK, kT, kSigma, kR, kQ);
    const double con = CashOrNothing(type, kS0, kK, kT, kSigma, kR, kQ);
    const double sign = type == OptionType::kCall ? 1.0 : -1.0;
    EXPECT_NEAR(sign * (aon - kK * con),
                ch03::BsPrice(type, kS0, kK, kT, kSigma, kR, kQ), 1e-13);
  }
  // Digital call + digital put = a zero coupon bond; AoN call + AoN put = the
  // asset.
  EXPECT_NEAR(CashOrNothing(OptionType::kCall, kS0, kK, kT, kSigma, kR, kQ) +
                  CashOrNothing(OptionType::kPut, kS0, kK, kT, kSigma, kR, kQ),
              std::exp(-kR * kT), 1e-15);
  EXPECT_NEAR(AssetOrNothing(OptionType::kCall, kS0, kK, kT, kSigma, kR, kQ) +
                  AssetOrNothing(OptionType::kPut, kS0, kK, kT, kSigma, kR, kQ),
              kS0 * std::exp(-kQ * kT), 1e-13);
}

TEST(Ch04InTheMoney, FinancialMeaningOfNd1Nd2) {
  const auto [n_d1, n_d2] =
      ComputeExerciseProbabilities(kS0, kK, kT, kSigma, kR, kQ);
  EXPECT_GT(n_d1, n_d2);
  // N(d2): risk-neutral exercise probability, the price of the digital grown at
  // r.
  EXPECT_NEAR(n_d2,
              std::exp(kR * kT) *
                  CashOrNothing(OptionType::kCall, kS0, kK, kT, kSigma, kR, kQ),
              1e-15);
  // N(d1) = E_RN[S_T 1{S_T > K}] / E_RN[S_T].
  const double fwd = kS0 * std::exp((kR - kQ) * kT);
  EXPECT_NEAR(
      n_d1,
      std::exp(kR * kT) *
          AssetOrNothing(OptionType::kCall, kS0, kK, kT, kSigma, kR, kQ) / fwd,
      1e-14);
  // e^(-qT) N(d1) is Delta; -e^(-rT) N(d2) is dC/dK.
  const auto g = ch03::BsGreeks(OptionType::kCall, kS0, kK, kT, kSigma, kR, kQ);
  EXPECT_NEAR(g.delta, std::exp(-kQ * kT) * n_d1, 1e-15);
  const double d_call_d_strike = p4mfe::ch01::Derivative(
      [](double k) { return ch03::BsCall(kS0, k, kT, kSigma, kR, kQ); }, kK);
  EXPECT_NEAR(d_call_d_strike, -std::exp(-kR * kT) * n_d2, 1e-8);
}
