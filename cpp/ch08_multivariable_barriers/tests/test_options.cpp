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
  const double S = 50, K = 52, T = 0.75, sigma = 0.3, r = 0.04, q = 0.01;
  for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
    const double exact = ch03::BsGreeks(type, S, K, T, sigma, r, q).theta;
    EXPECT_NEAR(theta_from_maturity(
                    [&](double tau) {
                      return ch03::BsPrice(type, S, K, tau, sigma, r, q);
                    },
                    T),
                exact, 1e-7);
    EXPECT_NEAR(
        theta_by_integral([&](double s) { return ch01::Payoff(type, s, K); }, S,
                          T, sigma, r, q),
        exact, 1e-4);
  }
  // Payoff S_T^2: V(tau) = e^(-r tau) E[S_T^2] = S^2 e^((r - 2q + sigma^2)
  // tau), so Theta = -(r - 2q + sigma^2) V.
  const auto sq = [](double s) { return s * s; };
  const double V = S * S * std::exp((r - 2 * q + sigma * sigma) * T);
  EXPECT_NEAR(theta_by_integral(sq, S, T, sigma, r, q),
              -(r - 2 * q + sigma * sigma) * V, 1e-4 * V);
}

// ---------------------------------------------------------------- 8.7 -----
TEST(Ch08BoxMuller, Transform) {
  const auto [z1, z2] =
      box_muller(std::exp(-2.0), 0.125);  // radius 2, angle pi/4
  EXPECT_NEAR(z1, std::numbers::sqrt2, 1e-15);
  EXPECT_NEAR(z2, std::numbers::sqrt2, 1e-15);
  const auto [u1, u2] = box_muller_inverse(-0.7, 1.3);
  const auto [w1, w2] = box_muller(u1, u2);
  EXPECT_NEAR(w1, -0.7, 1e-14);
  EXPECT_NEAR(w2, 1.3, 1e-14);
  // |det d(u1,u2)/d(z1,z2)| = phi(z1) phi(z2): the change of variables proof.
  for (const auto& z :
       {Point2{0.3, -1.2}, Point2{1.5, 0.4}, Point2{-0.2, -0.9}}) {
    const auto inv = [](double a, double b) {
      const auto [x, y] = box_muller_inverse(a, b);
      return Point2{x, y};
    };
    EXPECT_NEAR(jacobian_determinant(inv, z[0], z[1]),
                norm_pdf(z[0]) * norm_pdf(z[1]), 1e-9);
  }
}

TEST(Ch08BoxMuller, Samples) {
  const auto x = box_muller_samples(200'000, 11);
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
  EXPECT_NEAR(below / n, norm_cdf(-1.0), 0.003);
  EXPECT_NEAR(cross / (n / 2), 0.0, 0.01);  // independent pair
}

// ---------------------------------------------------------------- 8.8 -----
TEST(Ch08Heat, TransformSolvesHeatEquation) {
  const double K = 40, T = 0.5, sigma = 0.25, r = 0.05, q = 0.02;
  const auto c = heat_transform(sigma, r, q);
  // u = V / (K e^(ax + b tau)) satisfies u_tau = u_xx for V = Black-Scholes
  // call.
  const auto u = [&](double x, double tau) {
    const double theta = 2 * tau / (sigma * sigma);
    return heat_from_bs(ch03::BsCall(K * std::exp(x), K, theta, sigma, r, q), x,
                        tau, K, c);
  };
  for (double x : {-0.2, 0.0, 0.3}) {
    const double tau = 0.01;
    const double ut = ch01::Derivative([&](double s) { return u(x, s); }, tau);
    const double uxx =
        ch01::SecondDerivative([&](double y) { return u(y, tau); }, x);
    EXPECT_NEAR(ut, uxx, 1e-6 * std::max(1.0, std::abs(ut)));
  }
  // Round trip and the heat-kernel solution give Black-Scholes back.
  const auto [x, tau] = to_heat_variables(44, K, T, sigma);
  EXPECT_NEAR(bs_from_heat(heat_from_bs(3.0, x, tau, K, c), x, tau, K, c), 3.0,
              1e-14);
  for (OptionType type : {OptionType::kCall, OptionType::kPut}) {
    const auto h = [&](double s) { return ch01::Payoff(type, s, K); };
    for (double S : {35.0, 40.0, 46.0})
      EXPECT_NEAR(bs_price_via_heat(h, S, K, T, sigma, r, q),
                  ch03::BsPrice(type, S, K, T, sigma, r, q), 1e-7);
  }
}

TEST(Ch08Heat, ExplicitFiniteDifferences) {
  const double K = 40, T = 0.5, sigma = 0.25, r = 0.05, q = 0.02;
  for (OptionType type : {OptionType::kCall, OptionType::kPut})
    for (double S : {35.0, 40.0, 46.0})
      EXPECT_NEAR(european_heat_fd(type, S, K, T, sigma, r, q),
                  ch03::BsPrice(type, S, K, T, sigma, r, q), 2e-3);
  // Converges as the grid is refined (O(dx^2) on average; the payoff kink
  // makes individual errors irregular).
  const double exact = ch03::BsCall(46, K, T, sigma, r, q);
  const double e1 = std::abs(
      european_heat_fd(OptionType::kCall, 46, K, T, sigma, r, q, 100) - exact);
  const double e2 = std::abs(
      european_heat_fd(OptionType::kCall, 46, K, T, sigma, r, q, 800) - exact);
  EXPECT_GT(e1 / e2, 20.0);
  // alpha > 1/2 is unstable.
  const auto g = solve_heat_explicit(
      [](double x) { return std::max(x, 0.0); }, [](double) { return 0.0; },
      [](double) { return 1.0; }, -1, 1, 0.1, 40, 20);
  EXPECT_GT(g.alpha, 0.5);
  EXPECT_GT(std::abs(g.u[20]), 1e3);
}

// ---------------------------------------------------------------- 8.9 -----
TEST(Ch08Barrier, DownAndOutCall) {
  const double K = 100, T = 1, sigma = 0.3, r = 0.05, q = 0.02;
  for (double B : {80.0, 95.0, 110.0}) {
    for (double S : {B + 1, 105.0, 130.0}) {
      if (S <= B) continue;
      const double out = down_and_out_call(S, K, B, T, sigma, r, q);
      const double vanilla = ch03::BsCall(S, K, T, sigma, r, q);
      EXPECT_GT(out, 0.0);
      EXPECT_LT(out, vanilla);
      EXPECT_NEAR(out + down_and_in_call(S, K, B, T, sigma, r, q), vanilla,
                  1e-12);
      // Solves the Black-Scholes PDE (chapter 7).
      const auto V = [&](double s, double t) {
        return down_and_out_call(s, K, B, T - t, sigma, r, q);
      };
      EXPECT_NEAR(ch07::bs_pde_residual_fd(V, S, 0.0, sigma, r, q), 0.0, 1e-5)
          << B << ' ' << S;
      // Matches the heat equation solved on [B, S_max] with zero value at the
      // barrier.
      const double S_max = K * std::exp(6 * sigma);
      const double fd = bs_heat_fd(
          [&](double s) { return s > B ? std::max(s - K, 0.0) : 0.0; },
          [](double) { return 0.0; },
          [&](double th) {
            return S_max * std::exp(-q * th) - K * std::exp(-r * th);
          },
          S, K, T, sigma, r, q, B, S_max, 600);
      EXPECT_NEAR(fd, out, 5e-3) << B << ' ' << S;
    }
    EXPECT_EQ(down_and_out_call(B, K, B, T, sigma, r, q), 0.0);
    EXPECT_NEAR(down_and_out_call(B * (1 + 1e-9), K, B, T, sigma, r, q), 0.0,
                1e-6);
  }
  // Far barrier: knock-out ~ vanilla.
  EXPECT_NEAR(down_and_out_call(100, K, 20, T, sigma, r, q),
              ch03::BsCall(100, K, T, sigma, r, q), 1e-9);
}

TEST(Ch08Barrier, UpAndOutPutAndImage) {
  const double K = 100, T = 0.5, sigma = 0.25, r = 0.03, q = 0.0;
  for (double B : {90.0, 105.0, 130.0}) {
    const double S = B - 5;
    const double out = up_and_out_put(S, K, B, T, sigma, r, q);
    EXPECT_GT(out, 0.0);
    EXPECT_NEAR(out + up_and_in_put(S, K, B, T, sigma, r, q),
                ch03::BsPut(S, K, T, sigma, r, q), 1e-12);
    EXPECT_NEAR(up_and_out_put(B * (1 - 1e-9), K, B, T, sigma, r, q), 0.0,
                1e-6);
    const auto V = [&](double s, double t) {
      return up_and_out_put(s, K, B, T - t, sigma, r, q);
    };
    EXPECT_NEAR(ch07::bs_pde_residual_fd(V, S, 0.0, sigma, r, q), 0.0, 1e-5);
  }
  // The image of a solution is a solution, and equals it on S = B.
  const double B = 90;
  const auto call = [&](double s, double t) {
    return ch03::BsCall(s, K, T - t, sigma, r, 0.01);
  };
  const auto img = [&](double s, double t) {
    return barrier_image([&](double x) { return call(x, t); }, s, B, sigma, r,
                         0.01);
  };
  EXPECT_NEAR(ch07::bs_pde_residual_fd(img, 110, 0.1, sigma, r, 0.01), 0.0,
              1e-6);
  EXPECT_NEAR(img(B, 0.0), call(B, 0.0), 1e-12);
}

// --------------------------------------------------------------- 8.10 -----
TEST(Ch08EarlyExercise, CallWithoutDividends) {
  const double S = 100, K = 95, T = 1, sigma = 0.2, r = 0.05;
  // Same value American and European on the tree; the tree converges to
  // Black-Scholes.
  EXPECT_NEAR(early_exercise_premium(OptionType::kCall, S, K, T, sigma, r, 0.0),
              0.0, 1e-12);
  EXPECT_NEAR(binomial_price(OptionType::kCall, S, K, T, sigma, r, 0, 1000,
                             ExerciseStyle::European),
              ch03::BsCall(S, K, T, sigma, r), 2e-2);
  // Exercising gives up at least K (1 - e^(-r tau)).
  for (double tau : {0.1, 0.5, 2.0})
    for (double s : {80.0, 100.0, 200.0})
      EXPECT_GE(call_exercise_loss(s, K, tau, sigma, r),
                call_exercise_loss_bound(K, tau, r) - 1e-12);
  // But with dividends, or for puts, early exercise has value.
  EXPECT_GT(
      early_exercise_premium(OptionType::kCall, 150, K, T, sigma, r, 0.08),
      0.1);
  EXPECT_GT(early_exercise_premium(OptionType::kPut, 80, K, T, sigma, r, 0.0),
            0.1);
  EXPECT_THROW((void)binomial_price(OptionType::kCall, S, K, T, 0.01, 0.5, 0, 2,
                                    ExerciseStyle::European),
               std::invalid_argument);
}
