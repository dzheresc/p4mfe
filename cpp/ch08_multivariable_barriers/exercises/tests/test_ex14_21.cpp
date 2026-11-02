#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch08/exercises/ex14_itm_puts.hpp>
#include <p4mfe/ch08/exercises/ex15_time_value.hpp>
#include <p4mfe/ch08/exercises/ex16_17_down_and_out_call.hpp>
#include <p4mfe/ch08/exercises/ex18_doc_greeks.hpp>
#include <p4mfe/ch08/exercises/ex19_20_early_exercise.hpp>
#include <p4mfe/ch08/exercises/ex21_implied_vol_skew.hpp>

using namespace p4mfe;
using namespace p4mfe::ch08::ex;
using ch03::OptionType;

TEST(Ch08Ex14, ItmPutTable) {
  const auto rows = Ex14Table();
  const double values[] = {5.819577, 5.340299,  8.032528,
                           7.823400, 10.486171, 10.547645};
  const bool larger[] = {false, false, false, true, true, true};
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(rows[i].value, values[i], 1e-6) << i;
    EXPECT_EQ(rows[i].intrinsic_larger, larger[i]) << i;
    // Where intrinsic > European, the American put (binomial) is worth more.
    if (larger[i]) {
      const double am = ch08::BinomialPrice(
          OptionType::kPut, kEx14S, rows[i].strike, rows[i].maturity,
          kEx14Sigma, kEx14R, 0.0, 500, ch08::ExerciseStyle::kAmerican);
      EXPECT_GT(am, rows[i].value);
      EXPECT_GE(am, rows[i].intrinsic - 1e-12);
    }
  }
}

TEST(Ch08Ex15, TimeValueLargestAtTheMoney) {
  for (const Ex15Market m :
       {Ex15Market{}, Ex15Market{100.0, 2.0, 0.2, 0.03, 0.02},
        Ex15Market{50.0, 0.1, 0.5, 0.0, 0.0}}) {
    const double at_strike = Ex15TimeValue(m.strike, m);
    EXPECT_NEAR(at_strike,
                ch03::BsCall(m.strike, m.strike, m.maturity, m.sigma, m.r, m.q),
                1e-14);
    for (double spot = 0.2 * m.strike; spot < 5.0 * m.strike;
         spot += 0.005 * m.strike) {
      if (std::abs(spot - m.strike) < 1e-9) continue;
      EXPECT_LT(Ex15TimeValue(spot, m), at_strike) << spot;
      EXPECT_EQ(Ex15Derivative(spot, m) > 0.0, spot < m.strike) << spot;
    }
  }
}

TEST(Ch08Ex16, DownAndOutCall) {
  const Ex16Market m;
  EXPECT_NEAR(Ex16Formula846(m), 3.398883, 1e-6);
  EXPECT_NEAR(Ex16Value(m), Ex16Formula846(m), 1e-12);
  EXPECT_NEAR(ch08::DownAndInCall(m.spot, m.strike, m.barrier, m.maturity,
                                  m.sigma, m.r),
              0.455067, 1e-6);
}

TEST(Ch08Ex17, BarrierToZero) {
  const double spot = 40.0, strike = 40.0, maturity = 2.0, sigma = 0.6;
  // The hint's form of (8.46) with r = q = 0.
  for (double barrier : {5.0, 20.0, 35.0})
    EXPECT_NEAR(
        ch08::DownAndOutCall(spot, strike, barrier, maturity, sigma, 0.0),
        ch03::BsCall(spot, strike, maturity, sigma, 0.0) -
            Ex17Bracket(spot, strike, barrier, maturity, sigma),
        1e-12);
  // The bracket and N(d2)/B decrease to 0 as B -> 0.
  double prev_bracket = 1e300, prev_ratio = 1e300;
  for (double barrier = 35.0; barrier > 1e-3; barrier *= 0.7) {
    const double br = Ex17Bracket(spot, strike, barrier, maturity, sigma),
                 ratio = Ex17ND2OverB(spot, strike, barrier, maturity, sigma);
    EXPECT_LE(std::abs(br), prev_bracket) << barrier;
    EXPECT_LE(ratio, prev_ratio) << barrier;
    prev_bracket = std::abs(br);
    prev_ratio = ratio;
  }
  EXPECT_LT(prev_bracket, 1e-12);
  EXPECT_LT(prev_ratio, 1e-12);
}

TEST(Ch08Ex18, DeltaGammaOfDownAndOut) {
  const auto g = Ex18Greeks(40.0, 40.0, 35.0, 0.5, 0.3, 0.05);
  EXPECT_NEAR(g.delta, 0.703253, 1e-6);
  EXPECT_NEAR(g.gamma, 0.018758, 1e-6);
  for (double q : {0.0, 0.03})
    for (double spot : {35.5, 36.0, 38.0, 40.0, 45.0, 60.0}) {
      const auto a = Ex18Greeks(spot, 40.0, 35.0, 0.5, 0.3, 0.05, q);
      const auto f = Ex18GreeksFd(spot, 40.0, 35.0, 0.5, 0.3, 0.05, q);
      EXPECT_NEAR(a.delta, f.delta, 1e-6) << spot;
      EXPECT_NEAR(a.gamma, f.gamma, 1e-5) << spot;
    }
  EXPECT_LT(Ex18Greeks(36.0, 40.0, 35.0, 0.5, 0.3, 0.05).gamma, 0.0);
}

TEST(Ch08Ex19, CallAboveIntrinsicAndNoEarlyExercise) {
  for (double maturity : {0.05, 1.0, 5.0})
    for (double sigma : {0.05, 0.3, 0.8}) {
      const CallMarket m{40.0, maturity, sigma, 0.05, 0.0};
      // The Put-Call parity bound K(1 - e^(-rT)) is a uniform lower bound
      // for the time value; deep in the money the value is close to it.
      for (double spot = 1.0; spot < 400.0; spot *= 1.1) {
        // Far out of the money the value underflows to 0 in double precision.
        const double tv = Ex19TimeValue(spot, m);
        EXPECT_GE(tv, 0.0) << spot;
        if (spot > 0.8 * m.strike) {
          EXPECT_GT(tv, 0.0) << spot;
        }
        if (spot > m.strike) {
          EXPECT_GE(tv, Ex19LowerBoundGap(m) * (1.0 - 1e-9)) << spot;
        }
      }
    }
  for (double spot : {30.0, 40.0, 60.0}) {
    const double prem = ch08::EarlyExercisePremium(
        OptionType::kCall, spot, 40.0, 1.0, 0.3, 0.05, 0.0, 400);
    EXPECT_NEAR(prem, 0.0, 1e-10) << spot;
  }
}

TEST(Ch08Ex20, DividendCallBelowIntrinsic) {
  for (double q : {0.05, 0.01, 0.001}) {
    const CallMarket m{40.0, 1.0, 0.3, 0.05, q};
    const double spot_threshold = Ex20Threshold(m),
                 spot_crossover = Ex20Crossover(m);
    EXPECT_GT(spot_crossover, m.strike);
    EXPECT_LT(spot_crossover, spot_threshold);
    EXPECT_LT(Ex19TimeValue(spot_threshold, m), 0.0);
    EXPECT_LT(Ex19TimeValue(1.01 * spot_crossover, m), 0.0);
    EXPECT_GT(Ex19TimeValue(0.99 * spot_crossover, m), 0.0);
  }
  EXPECT_NEAR(Ex20Crossover({40.0, 1.0, 0.3, 0.05, 0.01}), 196.0594, 1e-3);
  EXPECT_NEAR(Ex20Crossover({40.0, 1.0, 0.3, 0.05, 0.05}), 56.7112, 1e-3);
}

TEST(Ch08Ex21, ImpliedVolSlope) {
  const SmileMarket m;
  for (double strike : {70.0, 90.0, 100.0, 115.0, 140.0}) {
    // Exact C'(K) for prices from the smile: dC_BS/dK + vega * sigma'(K).
    const double s = Ex21Smile(strike, m);
    const auto [d1, d2] = ch03::BsD1D2(m.spot, strike, m.maturity, s, m.r, m.q);
    const double vega = ch03::BsGreeks(OptionType::kCall, m.spot, strike,
                                       m.maturity, s, m.r, m.q)
                            .vega;
    const double d_call = -std::exp(-m.r * m.maturity) * norm_cdf(d2) +
                          vega * Ex21SmileSlope(strike, m);
    (void)d1;
    EXPECT_NEAR(Ex21DsigmaDK(strike, s, d_call, m), Ex21SmileSlope(strike, m),
                1e-12);
    // From prices only (implied vol by Newton, C'(K) by differences).
    EXPECT_NEAR(Ex21SlopeFromPrices([&](double k) { return Ex21Price(k, m); },
                                    strike, m),
                Ex21SmileSlope(strike, m), 1e-7)
        << strike;
  }
  // A flat smile gives zero slope.
  const auto flat = [&](double k) {
    return ch03::BsCall(m.spot, k, m.maturity, 0.25, m.r, m.q);
  };
  EXPECT_NEAR(Ex21SlopeFromPrices(flat, 100.0, m), 0.0, 1e-7);
}
