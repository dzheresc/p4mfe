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
  const auto rows = ex14_table();
  const double values[] = {5.819577, 5.340299,  8.032528,
                           7.823400, 10.486171, 10.547645};
  const bool larger[] = {false, false, false, true, true, true};
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(rows[i].value, values[i], 1e-6) << i;
    EXPECT_EQ(rows[i].intrinsic_larger, larger[i]) << i;
    // Where intrinsic > European, the American put (binomial) is worth more.
    if (larger[i]) {
      const double am = ch08::binomial_price(
          OptionType::kPut, ex14_S, rows[i].K, rows[i].T, ex14_sigma, ex14_r,
          0.0, 500, ch08::ExerciseStyle::American);
      EXPECT_GT(am, rows[i].value);
      EXPECT_GE(am, rows[i].intrinsic - 1e-12);
    }
  }
}

TEST(Ch08Ex15, TimeValueLargestAtTheMoney) {
  for (const Ex15Market m :
       {Ex15Market{}, Ex15Market{100.0, 2.0, 0.2, 0.03, 0.02},
        Ex15Market{50.0, 0.1, 0.5, 0.0, 0.0}}) {
    const double at_K = ex15_time_value(m.K, m);
    EXPECT_NEAR(at_K, ch03::bs_call(m.K, m.K, m.T, m.sigma, m.r, m.q), 1e-14);
    for (double S = 0.2 * m.K; S < 5.0 * m.K; S += 0.005 * m.K) {
      if (std::abs(S - m.K) < 1e-9) continue;
      EXPECT_LT(ex15_time_value(S, m), at_K) << S;
      EXPECT_EQ(ex15_derivative(S, m) > 0.0, S < m.K) << S;
    }
  }
}

TEST(Ch08Ex16, DownAndOutCall) {
  const Ex16Market m;
  EXPECT_NEAR(ex16_formula_846(m), 3.398883, 1e-6);
  EXPECT_NEAR(ex16_value(m), ex16_formula_846(m), 1e-12);
  EXPECT_NEAR(ch08::down_and_in_call(m.S, m.K, m.B, m.T, m.sigma, m.r),
              0.455067, 1e-6);
}

TEST(Ch08Ex17, BarrierToZero) {
  const double S = 40.0, K = 40.0, T = 2.0, sigma = 0.6;
  // The hint's form of (8.46) with r = q = 0.
  for (double B : {5.0, 20.0, 35.0})
    EXPECT_NEAR(
        ch08::down_and_out_call(S, K, B, T, sigma, 0.0),
        ch03::bs_call(S, K, T, sigma, 0.0) - ex17_bracket(S, K, B, T, sigma),
        1e-12);
  // The bracket and N(d2)/B decrease to 0 as B -> 0.
  double prev_bracket = 1e300, prev_ratio = 1e300;
  for (double B = 35.0; B > 1e-3; B *= 0.7) {
    const double br = ex17_bracket(S, K, B, T, sigma),
                 ratio = ex17_n_d2_over_b(S, K, B, T, sigma);
    EXPECT_LE(std::abs(br), prev_bracket) << B;
    EXPECT_LE(ratio, prev_ratio) << B;
    prev_bracket = std::abs(br);
    prev_ratio = ratio;
  }
  EXPECT_LT(prev_bracket, 1e-12);
  EXPECT_LT(prev_ratio, 1e-12);
}

TEST(Ch08Ex18, DeltaGammaOfDownAndOut) {
  const auto g = ex18_greeks(40.0, 40.0, 35.0, 0.5, 0.3, 0.05);
  EXPECT_NEAR(g.delta, 0.703253, 1e-6);
  EXPECT_NEAR(g.gamma, 0.018758, 1e-6);
  for (double q : {0.0, 0.03})
    for (double S : {35.5, 36.0, 38.0, 40.0, 45.0, 60.0}) {
      const auto a = ex18_greeks(S, 40.0, 35.0, 0.5, 0.3, 0.05, q);
      const auto f = ex18_greeks_fd(S, 40.0, 35.0, 0.5, 0.3, 0.05, q);
      EXPECT_NEAR(a.delta, f.delta, 1e-6) << S;
      EXPECT_NEAR(a.gamma, f.gamma, 1e-5) << S;
    }
  EXPECT_LT(ex18_greeks(36.0, 40.0, 35.0, 0.5, 0.3, 0.05).gamma, 0.0);
}

TEST(Ch08Ex19, CallAboveIntrinsicAndNoEarlyExercise) {
  for (double T : {0.05, 1.0, 5.0})
    for (double sigma : {0.05, 0.3, 0.8}) {
      const CallMarket m{40.0, T, sigma, 0.05, 0.0};
      // The Put-Call parity bound K(1 - e^(-rT)) is a uniform lower bound
      // for the time value; deep in the money the value is close to it.
      for (double S = 1.0; S < 400.0; S *= 1.1) {
        // Far out of the money the value underflows to 0 in double precision.
        const double tv = ex19_time_value(S, m);
        EXPECT_GE(tv, 0.0) << S;
        if (S > 0.8 * m.K) {
          EXPECT_GT(tv, 0.0) << S;
        }
        if (S > m.K) {
          EXPECT_GE(tv, ex19_lower_bound_gap(m) * (1.0 - 1e-9)) << S;
        }
      }
    }
  for (double S : {30.0, 40.0, 60.0}) {
    const double prem = ch08::early_exercise_premium(OptionType::kCall, S, 40.0,
                                                     1.0, 0.3, 0.05, 0.0, 400);
    EXPECT_NEAR(prem, 0.0, 1e-10) << S;
  }
}

TEST(Ch08Ex20, DividendCallBelowIntrinsic) {
  for (double q : {0.05, 0.01, 0.001}) {
    const CallMarket m{40.0, 1.0, 0.3, 0.05, q};
    const double S0 = ex20_threshold(m), Sx = ex20_crossover(m);
    EXPECT_GT(Sx, m.K);
    EXPECT_LT(Sx, S0);
    EXPECT_LT(ex19_time_value(S0, m), 0.0);
    EXPECT_LT(ex19_time_value(1.01 * Sx, m), 0.0);
    EXPECT_GT(ex19_time_value(0.99 * Sx, m), 0.0);
  }
  EXPECT_NEAR(ex20_crossover({40.0, 1.0, 0.3, 0.05, 0.01}), 196.0594, 1e-3);
  EXPECT_NEAR(ex20_crossover({40.0, 1.0, 0.3, 0.05, 0.05}), 56.7112, 1e-3);
}

TEST(Ch08Ex21, ImpliedVolSlope) {
  const SmileMarket m;
  for (double K : {70.0, 90.0, 100.0, 115.0, 140.0}) {
    // Exact C'(K) for prices from the smile: dC_BS/dK + vega * sigma'(K).
    const double s = ex21_smile(K, m);
    const auto [d1, d2] = ch03::bs_d1_d2(m.S, K, m.T, s, m.r, m.q);
    const double vega =
        ch03::bs_greeks(OptionType::kCall, m.S, K, m.T, s, m.r, m.q).vega;
    const double dC =
        -std::exp(-m.r * m.T) * norm_cdf(d2) + vega * ex21_smile_slope(K, m);
    (void)d1;
    EXPECT_NEAR(ex21_dsigma_dK(K, s, dC, m), ex21_smile_slope(K, m), 1e-12);
    // From prices only (implied vol by Newton, C'(K) by differences).
    EXPECT_NEAR(ex21_slope_from_prices(
                    [&](double k) { return ex21_price(k, m); }, K, m),
                ex21_smile_slope(K, m), 1e-7)
        << K;
  }
  // A flat smile gives zero slope.
  const auto flat = [&](double k) {
    return ch03::bs_call(m.S, k, m.T, 0.25, m.r, m.q);
  };
  EXPECT_NEAR(ex21_slope_from_prices(flat, 100.0, m), 0.0, 1e-7);
}
