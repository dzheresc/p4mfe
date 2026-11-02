#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <numeric>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch04/in_the_money.hpp>
#include <p4mfe/ch07/bs_pde.hpp>
#include <p4mfe/ch08/barrier.hpp>
#include <p4mfe/ch08/box_muller.hpp>
#include <p4mfe/ch08/double_integrals.hpp>
#include <p4mfe/ch08/early_exercise.hpp>
#include <p4mfe/ch08/heat_equation.hpp>
#include <p4mfe/ch08/theta.hpp>

using namespace p4mfe;
using namespace p4mfe::ch08;

// ---------------------------------------------------------------- 8.5 -----
TEST(Ch08Theta, MatchesClosedForm) {
  const double spot = 50, strike = 52, maturity = 0.75, sigma = 0.3, r = 0.04,
               q = 0.01;
  for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
    const double exact =
        ch03::BsGreeks(type, spot, strike, maturity, sigma, r, q).theta;
    EXPECT_NEAR(ThetaFromMaturity(
                    [&](double tau) {
                      return ch03::BsPrice(type, spot, strike, tau, sigma, r,
                                           q);
                    },
                    maturity),
                exact, 1e-7);
    EXPECT_NEAR(
        ThetaByIntegral([&](double s) { return ch01::Payoff(type, s, strike); },
                        spot, maturity, sigma, r, q),
        exact, 1e-4);
  }
  // Payoff S_T^2: V(tau) = e^(-r tau) E[S_T^2] = S^2 e^((r - 2q + sigma^2)
  // tau), so Theta = -(r - 2q + sigma^2) V.
  const auto sq = [](double s) { return s * s; };
  const double value =
      spot * spot * std::exp((r - 2 * q + sigma * sigma) * maturity);
  EXPECT_NEAR(ThetaByIntegral(sq, spot, maturity, sigma, r, q),
              -(r - 2 * q + sigma * sigma) * value, 1e-4 * value);
}

// ---------------------------------------------------------------- 8.7 -----
TEST(Ch08BoxMuller, Transform) {
  const auto [z1, z2] =
      BoxMuller(std::exp(-2.0), 0.125);  // radius 2, angle pi/4
  EXPECT_NEAR(z1, std::numbers::sqrt2, 1e-15);
  EXPECT_NEAR(z2, std::numbers::sqrt2, 1e-15);
  const auto [u1, u2] = BoxMullerInverse(-0.7, 1.3);
  const auto [w1, w2] = BoxMuller(u1, u2);
  EXPECT_NEAR(w1, -0.7, 1e-14);
  EXPECT_NEAR(w2, 1.3, 1e-14);
  // |det d(u1,u2)/d(z1,z2)| = phi(z1) phi(z2): the change of variables proof.
  for (const auto& z :
       {Point2{0.3, -1.2}, Point2{1.5, 0.4}, Point2{-0.2, -0.9}}) {
    const auto inv = [](double a, double b) {
      const auto [x, y] = BoxMullerInverse(a, b);
      return Point2{x, y};
    };
    EXPECT_NEAR(JacobianDeterminant(inv, z[0], z[1]),
                NormPdf(z[0]) * NormPdf(z[1]), 1e-9);
  }
}

TEST(Ch08BoxMuller, Samples) {
  const auto x = BoxMullerSamples(200'000, 11);
  const double n = static_cast<double>(x.size());
  const double m = std::accumulate(x.begin(), x.end(), 0.0) / n;
  double m2 = 0, m4 = 0, below = 0, cross = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    m2 += x[i] * x[i];
    m4 += std::pow(x[i], 4);
    below += x[i] < -1.0;
    if (i % 2 == 0) cross += x[i] * x[i + 1];  // the pair from one draw
  }
  EXPECT_NEAR(m, 0.0, 0.01);
  EXPECT_NEAR(m2 / n, 1.0, 0.01);
  EXPECT_NEAR(m4 / n, 3.0, 0.05);
  EXPECT_NEAR(below / n, NormCdf(-1.0), 0.003);
  EXPECT_NEAR(cross / (n / 2), 0.0, 0.01);  // independent pair
}

// ---------------------------------------------------------------- 8.8 -----
TEST(Ch08Heat, TransformSolvesHeatEquation) {
  const double strike = 40, maturity = 0.5, sigma = 0.25, r = 0.05, q = 0.02;
  const auto c = MakeHeatTransform(sigma, r, q);
  // u = V / (K e^(ax + b tau)) satisfies u_tau = u_xx for V = Black-Scholes
  // call.
  const auto u = [&](double x, double tau) {
    const double theta = 2 * tau / (sigma * sigma);
    return HeatFromBs(
        ch03::BsCall(strike * std::exp(x), strike, theta, sigma, r, q), x, tau,
        strike, c);
  };
  for (double x : {-0.2, 0.0, 0.3}) {
    const double tau = 0.01;
    const double ut = ch01::Derivative([&](double s) { return u(x, s); }, tau);
    const double uxx =
        ch01::SecondDerivative([&](double y) { return u(y, tau); }, x);
    EXPECT_NEAR(ut, uxx, 1e-6 * std::max(1.0, std::abs(ut)));
  }
  // Round trip and the heat-kernel solution give Black-Scholes back.
  const auto [x, tau] = ToHeatVariables(44, strike, maturity, sigma);
  EXPECT_NEAR(BsFromHeat(HeatFromBs(3.0, x, tau, strike, c), x, tau, strike, c),
              3.0, 1e-14);
  for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
    const auto h = [&](double s) { return ch01::Payoff(type, s, strike); };
    for (double spot : {35.0, 40.0, 46.0})
      EXPECT_NEAR(BsPriceViaHeat(h, spot, strike, maturity, sigma, r, q),
                  ch03::BsPrice(type, spot, strike, maturity, sigma, r, q),
                  1e-7);
  }
}

TEST(Ch08Heat, ExplicitFiniteDifferences) {
  const double strike = 40, maturity = 0.5, sigma = 0.25, r = 0.05, q = 0.02;
  for (OptionType type : {OptionType::kCall, OptionType::kPut})
    for (double spot : {35.0, 40.0, 46.0})
      EXPECT_NEAR(EuropeanHeatFd(type, spot, strike, maturity, sigma, r, q),
                  ch03::BsPrice(type, spot, strike, maturity, sigma, r, q),
                  2e-3);
  // Converges as the grid is refined (O(dx^2) on average; the payoff kink
  // makes individual errors irregular).
  const double exact = ch03::BsCall(46, strike, maturity, sigma, r, q);
  const double e1 = std::abs(EuropeanHeatFd(OptionType::kCall, 46, strike,
                                            maturity, sigma, r, q, 100) -
                             exact);
  const double e2 = std::abs(EuropeanHeatFd(OptionType::kCall, 46, strike,
                                            maturity, sigma, r, q, 800) -
                             exact);
  EXPECT_GT(e1 / e2, 20.0);
  // alpha > 1/2 is unstable.
  const auto g = SolveHeatExplicit(
      [](double x) { return std::max(x, 0.0); }, [](double) { return 0.0; },
      [](double) { return 1.0; }, -1, 1, 0.1, 40, 20);
  EXPECT_GT(g.alpha, 0.5);
  EXPECT_GT(std::abs(g.u[20]), 1e3);
}

TEST(Ch08Heat, CrankNicolsonAgreesWithExplicit) {
  const double strike = 40, maturity = 0.5, sigma = 0.25, r = 0.05, q = 0.02;
  for (OptionType type : {OptionType::kCall, OptionType::kPut})
    for (double spot : {35.0, 40.0, 46.0}) {
      const double cn =
          EuropeanHeatCrankNicolson(type, spot, strike, maturity, sigma, r, q);
      EXPECT_NEAR(cn, ch03::BsPrice(type, spot, strike, maturity, sigma, r, q),
                  2e-3);
      EXPECT_NEAR(cn, EuropeanHeatFd(type, spot, strike, maturity, sigma, r, q),
                  2e-3);
    }
  // Converges as O(dx^2) as the grid is refined. At the money the kink sits
  // on a grid node and the error is regular (a factor of 4 per halving of dx).
  const double exact = ch03::BsCall(strike, strike, maturity, sigma, r, q);
  const auto error = [&](std::size_t nx) {
    return std::abs(EuropeanHeatCrankNicolson(OptionType::kCall, strike, strike,
                                              maturity, sigma, r, q, nx) -
                    exact);
  };
  EXPECT_GT(error(100) / error(800), 40.0);
}

TEST(Ch08Heat, CrankNicolsonGridMatchesExplicitGrid) {
  // u0 = sin(pi x) on [0, 1] with zero boundaries: u = e^(-pi^2 tau) sin(pi x).
  const double pi = std::numbers::pi, tau_final = 0.05;
  const auto u0 = [&](double x) { return std::sin(pi * x); };
  const auto zero = [](double) { return 0.0; };
  const auto ex = SolveHeatExplicit(u0, zero, zero, 0, 1, tau_final, 50, 500);
  const auto cn =
      SolveHeatCrankNicolson(u0, zero, zero, 0, 1, tau_final, 50, 50);
  ASSERT_EQ(cn.x, ex.x);
  for (std::size_t i = 0; i < cn.x.size(); ++i) {
    const double exact = std::exp(-pi * pi * tau_final) * u0(cn.x[i]);
    EXPECT_NEAR(cn.u[i], exact, 2e-4);
    EXPECT_NEAR(cn.u[i], ex.u[i], 2e-4);
  }
  // The grid on which the explicit scheme blows up (alpha > 1/2) stays bounded.
  const auto g =
      SolveHeatCrankNicolson([](double x) { return std::max(x, 0.0); }, zero,
                             [](double) { return 1.0; }, -1, 1, 0.1, 40, 20);
  EXPECT_GT(g.alpha, 0.5);
  for (double u : g.u) {
    EXPECT_GE(u, -1e-12);
    EXPECT_LE(u, 1.0 + 1e-12);
  }
  EXPECT_THROW((void)SolveHeatCrankNicolson(u0, zero, zero, 0, 1, 0.1, 1, 10),
               std::invalid_argument);
}

// ---------------------------------------------------------------- 8.9 -----
TEST(Ch08Barrier, DownAndOutCall) {
  const double strike = 100, maturity = 1, sigma = 0.3, r = 0.05, q = 0.02;
  for (double barrier : {80.0, 95.0, 110.0}) {
    for (double spot : {barrier + 1, 105.0, 130.0}) {
      if (spot <= barrier) continue;
      const double out =
          DownAndOutCall(spot, strike, barrier, maturity, sigma, r, q);
      const double vanilla = ch03::BsCall(spot, strike, maturity, sigma, r, q);
      EXPECT_GT(out, 0.0);
      EXPECT_LT(out, vanilla);
      EXPECT_NEAR(
          out + DownAndInCall(spot, strike, barrier, maturity, sigma, r, q),
          vanilla, 1e-12);
      // Solves the Black-Scholes PDE (chapter 7).
      const auto value = [&](double s, double t) {
        return DownAndOutCall(s, strike, barrier, maturity - t, sigma, r, q);
      };
      EXPECT_NEAR(ch07::BsPdeResidualFd(value, spot, 0.0, sigma, r, q), 0.0,
                  1e-5)
          << barrier << ' ' << spot;
      // Matches the heat equation solved on [B, S_max] with zero value at the
      // barrier.
      const double spot_max = strike * std::exp(6 * sigma);
      const double fd = BsHeatFd(
          [&](double s) {
            return s > barrier ? std::max(s - strike, 0.0) : 0.0;
          },
          [](double) { return 0.0; },
          [&](double th) {
            return spot_max * std::exp(-q * th) - strike * std::exp(-r * th);
          },
          spot, strike, maturity, sigma, r, q, barrier, spot_max, 600);
      EXPECT_NEAR(fd, out, 5e-3) << barrier << ' ' << spot;
      const double cn = BsHeatCrankNicolson(
          [&](double s) {
            return s > barrier ? std::max(s - strike, 0.0) : 0.0;
          },
          [](double) { return 0.0; },
          [&](double th) {
            return spot_max * std::exp(-q * th) - strike * std::exp(-r * th);
          },
          spot, strike, maturity, sigma, r, q, barrier, spot_max, 600);
      EXPECT_NEAR(cn, out, 5e-3) << barrier << ' ' << spot;
      EXPECT_NEAR(cn, fd, 5e-3) << barrier << ' ' << spot;
    }
    EXPECT_EQ(DownAndOutCall(barrier, strike, barrier, maturity, sigma, r, q),
              0.0);
    EXPECT_NEAR(DownAndOutCall(barrier * (1 + 1e-9), strike, barrier, maturity,
                               sigma, r, q),
                0.0, 1e-6);
  }
  // Far barrier: knock-out ~ vanilla.
  EXPECT_NEAR(DownAndOutCall(100, strike, 20, maturity, sigma, r, q),
              ch03::BsCall(100, strike, maturity, sigma, r, q), 1e-9);
}

TEST(Ch08Barrier, UpAndOutPutAndImage) {
  const double strike = 100, maturity = 0.5, sigma = 0.25, r = 0.03, q = 0.0;
  for (double barrier : {90.0, 105.0, 130.0}) {
    const double spot = barrier - 5;
    const double out =
        UpAndOutPut(spot, strike, barrier, maturity, sigma, r, q);
    EXPECT_GT(out, 0.0);
    EXPECT_NEAR(out + UpAndInPut(spot, strike, barrier, maturity, sigma, r, q),
                ch03::BsPut(spot, strike, maturity, sigma, r, q), 1e-12);
    EXPECT_NEAR(UpAndOutPut(barrier * (1 - 1e-9), strike, barrier, maturity,
                            sigma, r, q),
                0.0, 1e-6);
    const auto value = [&](double s, double t) {
      return UpAndOutPut(s, strike, barrier, maturity - t, sigma, r, q);
    };
    EXPECT_NEAR(ch07::BsPdeResidualFd(value, spot, 0.0, sigma, r, q), 0.0,
                1e-5);
  }
  // The image of a solution is a solution, and equals it on S = B.
  const double barrier = 90;
  const auto call = [&](double s, double t) {
    return ch03::BsCall(s, strike, maturity - t, sigma, r, 0.01);
  };
  const auto img = [&](double s, double t) {
    return BarrierImage([&](double x) { return call(x, t); }, s, barrier, sigma,
                        r, 0.01);
  };
  EXPECT_NEAR(ch07::BsPdeResidualFd(img, 110, 0.1, sigma, r, 0.01), 0.0, 1e-6);
  EXPECT_NEAR(img(barrier, 0.0), call(barrier, 0.0), 1e-12);
}

// --------------------------------------------------------------- 8.10 -----
TEST(Ch08EarlyExercise, CallWithoutDividends) {
  const double spot = 100, strike = 95, maturity = 1, sigma = 0.2, r = 0.05;
  // Same value American and European on the tree; the tree converges to
  // Black-Scholes.
  EXPECT_NEAR(EarlyExercisePremium(OptionType::kCall, spot, strike, maturity,
                                   sigma, r, 0.0),
              0.0, 1e-12);
  EXPECT_NEAR(BinomialPrice(OptionType::kCall, spot, strike, maturity, sigma, r,
                            0, 1000, ExerciseStyle::kEuropean),
              ch03::BsCall(spot, strike, maturity, sigma, r), 2e-2);
  // Exercising gives up at least K (1 - e^(-r tau)).
  for (double tau : {0.1, 0.5, 2.0})
    for (double s : {80.0, 100.0, 200.0})
      EXPECT_GE(CallExerciseLoss(s, strike, tau, sigma, r),
                CallExerciseLossBound(strike, tau, r) - 1e-12);
  // But with dividends, or for puts, early exercise has value.
  EXPECT_GT(EarlyExercisePremium(OptionType::kCall, 150, strike, maturity,
                                 sigma, r, 0.08),
            0.1);
  EXPECT_GT(EarlyExercisePremium(OptionType::kPut, 80, strike, maturity, sigma,
                                 r, 0.0),
            0.1);
  EXPECT_THROW((void)BinomialPrice(OptionType::kCall, spot, strike, maturity,
                                   0.01, 0.5, 0, 2, ExerciseStyle::kEuropean),
               std::invalid_argument);
}
