#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch03/exercises/ex12_15_black_scholes.hpp>
#include <p4mfe/ch03/exercises/ex16_21_greeks_strike.hpp>

using namespace p4mfe::ch03;
using namespace p4mfe::ch03::ex;

namespace {
struct Case {
  double S, K, T, sigma, r, q;
};
const Case cases[] = {{50, 45, 0.5, 0.2, 0.06, 0.02},
                      {100, 110, 1.5, 0.35, 0.03, 0.0},
                      {40, 38, 0.25, 0.15, 0.01, 0.04},
                      {75, 75, 3.0, 0.25, 0.05, 0.05}};
}  // namespace

TEST(Ch03Ex12, CallAndPut) {
  const auto [d1, d2] = bs_d1_d2(50, 45, 0.5, 0.2, 0.06, 0.02);
  EXPECT_NEAR(d1, 0.957143, 1e-6);
  EXPECT_NEAR(d2, 0.815722, 1e-6);
  EXPECT_NEAR(ex12_call(), 6.508365, 1e-6);
  EXPECT_NEAR(ex12_put(), 0.675922, 1e-6);
  EXPECT_NEAR(ex12_call() - ex12_put(),
              50 * std::exp(-0.01) - 45 * std::exp(-0.03), 1e-12);
}

TEST(Ch03Ex13, ZeroStrike) {
  EXPECT_EQ(ex13_put_zero_strike(), 0.0);
  // Black-Scholes as K -> 0 agrees with the model-free values.
  EXPECT_NEAR(bs_call(80, 1e-8, 2.0, 0.3, 0.05, 0.03),
              ex13_call_zero_strike(80, 2.0, 0.03), 1e-7);
  EXPECT_NEAR(bs_put(80, 1e-8, 2.0, 0.3, 0.05, 0.03), 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(ex13_call_zero_strike(80, 2.0), 80.0);
}

TEST(Ch03Ex14, PutRhoFromParity) {
  for (const auto& c : cases) {
    const double rho =
        bs_greeks(OptionType::Put, c.S, c.K, c.T, c.sigma, c.r, c.q).rho;
    EXPECT_NEAR(ex14_rho_put_from_parity(c.S, c.K, c.T, c.sigma, c.r, c.q), rho,
                1e-10);
    const double h = 1e-5;
    const double fd = (bs_put(c.S, c.K, c.T, c.sigma, c.r + h, c.q) -
                       bs_put(c.S, c.K, c.T, c.sigma, c.r - h, c.q)) /
                      (2 * h);
    EXPECT_NEAR(rho, fd, 1e-5);
  }
}

TEST(Ch03Ex15, AssetOrNothingDelta) {
  for (const auto& c : cases) {
    const double sigma = 0.3, h = 1e-4;
    const double fd = (ex15_aon_call(c.S + h, c.K, c.T, sigma, c.r, c.q) -
                       ex15_aon_call(c.S - h, c.K, c.T, sigma, c.r, c.q)) /
                      (2 * h);
    EXPECT_NEAR(ex15_aon_delta(c.S, c.K, c.T, sigma, c.r, c.q), fd, 1e-7);
    // Larger than the plain call's Delta.
    EXPECT_GT(
        ex15_aon_delta(c.S, c.K, c.T, sigma, c.r, c.q),
        bs_greeks(OptionType::Call, c.S, c.K, c.T, sigma, c.r, c.q).delta);
  }
}

TEST(Ch03Ex16, VolgaVanna) {
  for (const auto& c : cases) {
    const double h = 1e-4;
    const auto vega = [&](OptionType t, double S, double sg) {
      return bs_greeks(t, S, c.K, c.T, sg, c.r, c.q).vega;
    };
    const auto delta = [&](OptionType t, double sg) {
      return bs_greeks(t, c.S, c.K, c.T, sg, c.r, c.q).delta;
    };
    for (auto t : {OptionType::Call, OptionType::Put}) {
      const double volga_fd =
          (vega(t, c.S, c.sigma + h) - vega(t, c.S, c.sigma - h)) / (2 * h);
      const double vanna_fd =
          (vega(t, c.S + h, c.sigma) - vega(t, c.S - h, c.sigma)) / (2 * h);
      const double vanna_fd2 =
          (delta(t, c.sigma + h) - delta(t, c.sigma - h)) / (2 * h);
      EXPECT_NEAR(ex16_volga(c.S, c.K, c.T, c.sigma, c.r, c.q), volga_fd, 1e-5);
      EXPECT_NEAR(ex16_vanna(c.S, c.K, c.T, c.sigma, c.r, c.q), vanna_fd, 1e-6);
      EXPECT_NEAR(ex16_vanna(c.S, c.K, c.T, c.sigma, c.r, c.q), vanna_fd2,
                  1e-6);
    }
  }
}

TEST(Ch03Ex17, AtmCallVersusPut) {
  for (double r : {0.01, 0.04, 0.08})
    for (double q : {0.0, 0.04, 0.1}) {
      const double diff = ex17_atm_call_minus_put(60, 1.5, 0.25, r, q);
      EXPECT_NEAR(diff, 60 * (std::exp(-q * 1.5) - std::exp(-r * 1.5)), 1e-11);
      if (q < r) {
        EXPECT_GT(diff, 0.0);
      }
      if (q > r) {
        EXPECT_LT(diff, 0.0);
      }
      if (q == r) {
        EXPECT_NEAR(diff, 0.0, 1e-11);
      }
    }
}

TEST(Ch03Ex18, ThetaSign) {
  for (double S : {20.0, 50.0, 80.0})
    for (double T : {0.01, 0.5, 5.0, 50.0})
      for (double r : {0.0, 0.03, 0.1})
        EXPECT_LT(ex18_theta_call(S, 50, T, 0.3, r), 0.0);
  // q = r ATM: negative for short, positive for long maturities.
  const double Tstar = ex18_theta_sign_change(0.2, 0.05);
  EXPECT_NEAR(Tstar, 9.6814, 1e-4);
  EXPECT_LT(ex18_theta_call(100, 100, 1.0, 0.2, 0.05, 0.05), 0.0);
  EXPECT_GT(ex18_theta_call(100, 100, 10.0, 0.2, 0.05, 0.05), 0.0);
  EXPECT_NEAR(ex18_theta_call(100, 100, 10.0, 0.2, 0.05, 0.05), 0.024754, 1e-6);
  for (double T : {0.5, 5.0, 20.0})
    EXPECT_NEAR(
        ex18_theta_call(100, 100, T, 0.2, 0.05, 0.05),
        100 * std::exp(-0.05 * T) * ex18_atm_q_eq_r_bracket(T, 0.2, 0.05),
        1e-11);
}

TEST(Ch03Ex19_20, StrikeDerivatives) {
  for (const auto& c : cases) {
    const double h = 1e-3;
    const auto C = [&](double K) {
      return bs_call(c.S, K, c.T, c.sigma, c.r, c.q);
    };
    const auto P = [&](double K) {
      return bs_put(c.S, K, c.T, c.sigma, c.r, c.q);
    };
    const auto d = ex19_strike_derivatives(c.S, c.K, c.T, c.sigma, c.r, c.q);
    EXPECT_NEAR(d.dC_dK, (C(c.K + h) - C(c.K - h)) / (2 * h), 1e-8);
    EXPECT_NEAR(d.dP_dK, (P(c.K + h) - P(c.K - h)) / (2 * h), 1e-8);
    EXPECT_NEAR(d.d2C_dK2, (C(c.K + h) - 2 * C(c.K) + C(c.K - h)) / (h * h),
                1e-5);
    EXPECT_NEAR(d.d2P_dK2, (P(c.K + h) - 2 * P(c.K) + P(c.K - h)) / (h * h),
                1e-5);
    EXPECT_GT(d.d2C_dK2, 0.0);  // exercise 20
    EXPECT_NEAR(d.dP_dK - d.dC_dK, std::exp(-c.r * c.T), 1e-14);
  }
}

TEST(Ch03Ex21, MinimalStraddle) {
  for (const auto& c : cases) {
    const double Ks = ex21_min_straddle_strike(c.S, c.T, c.sigma, c.r, c.q);
    EXPECT_NEAR(bs_d1_d2(c.S, Ks, c.T, c.sigma, c.r, c.q).d2, 0.0, 1e-12);
    const double f = ex21_straddle(c.S, Ks, c.T, c.sigma, c.r, c.q);
    for (double dK : {-1.0, -0.1, 0.1, 1.0})
      EXPECT_GT(ex21_straddle(c.S, Ks + dK, c.T, c.sigma, c.r, c.q), f);
    // Golden-section search lands on the same strike.
    double a = 0.5 * c.S, b = 1.5 * c.S;
    const double g = (std::sqrt(5.0) - 1) / 2;
    for (int i = 0; i < 100; ++i) {
      const double x1 = b - g * (b - a), x2 = a + g * (b - a);
      if (ex21_straddle(c.S, x1, c.T, c.sigma, c.r, c.q) <
          ex21_straddle(c.S, x2, c.T, c.sigma, c.r, c.q))
        b = x2;
      else
        a = x1;
    }
    EXPECT_NEAR(0.5 * (a + b), Ks, 1e-4 * c.S);
  }
}
