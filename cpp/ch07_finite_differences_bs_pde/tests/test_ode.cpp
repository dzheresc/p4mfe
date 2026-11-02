#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch07/ode.hpp>

using namespace p4mfe::ch07;

// ---------------------------------------------------------------- 7.2 -----
TEST(Ch07Ode, ForwardBackwardCentralConverge) {
  // y' = -2 x y, y(0) = 1: y = e^(-x^2).
  const auto f = [](double x, double y) { return -2 * x * y; };
  const auto exact = [](double x) { return std::exp(-x * x); };
  for (OdeScheme s :
       {OdeScheme::Forward, OdeScheme::Backward, OdeScheme::Central}) {
    const double e1 = max_error(solve_ivp(f, 0, 1, 2, 100, s), exact);
    const double e2 = max_error(solve_ivp(f, 0, 1, 2, 200, s), exact);
    const double order = std::log2(e1 / e2);
    EXPECT_NEAR(order, s == OdeScheme::Central ? 2.0 : 1.0, 0.15);
    EXPECT_LT(e2, 1e-2);
  }
  const auto sol = euler_forward(f, 0, 1, 2, 4);
  ASSERT_EQ(sol.x.size(), 5u);
  EXPECT_DOUBLE_EQ(sol.x[4], 2.0);
  EXPECT_DOUBLE_EQ(sol.y[1], 1.0);        // f(0, 1) = 0
  EXPECT_DOUBLE_EQ(sol.y[2], 1.0 - 0.5);  // y_1 + h * (-2 * 0.5 * 1)
}

TEST(Ch07Ode, BackwardEulerNonlinear) {
  // y' = y^2, y(0) = 1/2: y = 1 / (2 - x); each implicit step is a quadratic.
  const auto f = [](double, double y) { return y * y; };
  const auto sol = euler_backward(f, 0, 0.5, 1, 400);
  EXPECT_NEAR(sol.y.back(), 1.0, 2e-3);
  // The step equation holds.
  const double h = 1.0 / 400;
  for (std::size_t k = 0; k < 400; k += 57)
    EXPECT_NEAR(sol.y[k + 1] - sol.y[k] - h * f(0, sol.y[k + 1]), 0, 1e-12);
}

TEST(Ch07Ode, Stability) {
  // y' = -20 y, y(0) = 1 on [0, 3] with h = 0.15: h * 20 = 3 > 2.
  const auto f = [](double, double y) { return -20 * y; };
  EXPECT_GT(std::abs(euler_forward(f, 0, 1, 3, 20).y.back()),
            1e3);  // explicit blows up
  EXPECT_LT(std::abs(euler_backward(f, 0, 1, 3, 20).y.back()),
            1e-6);  // implicit decays
  // Leapfrog is unstable for decay at any h: the parasitic mode grows.
  EXPECT_GT(std::abs(leapfrog(f, 0, 1, 3, 600).y.back()), 1.0);
}

TEST(Ch07Ode, Tridiagonal) {
  // [2 1 0; 1 3 1; 0 1 4] x = [3 5 5] has x = (1, 1, 1).
  const auto x = solve_tridiagonal({0, 1, 1}, {2, 3, 4}, {1, 1, 0}, {3, 5, 5});
  for (double v : x) EXPECT_NEAR(v, 1.0, 1e-14);
  EXPECT_THROW((void)solve_tridiagonal({0}, {1, 2}, {0, 0}, {1, 1}),
               std::invalid_argument);
}

TEST(Ch07Ode, BoundaryValueProblem) {
  // y'' - y = 0 on [0, 1], y(0) = 0, y(1) = sinh(1): y = sinh(x).
  const auto zero = [](double) { return 0.0; };
  const auto minus_one = [](double) { return -1.0; };
  const auto e1 =
      max_error(solve_bvp(zero, minus_one, zero, 0, 1, 0, std::sinh(1.0), 20),
                [](double x) { return std::sinh(x); });
  const auto e2 =
      max_error(solve_bvp(zero, minus_one, zero, 0, 1, 0, std::sinh(1.0), 40),
                [](double x) { return std::sinh(x); });
  EXPECT_NEAR(std::log2(e1 / e2), 2.0, 0.05);
  EXPECT_LT(e2, 1e-5);

  // With a first-derivative term: y'' + 2 y' + y = e^x has solutions
  // e^x / 4 + (A + B x) e^(-x); take A = 0, B = 1.
  const auto exact = [](double x) {
    return std::exp(x) / 4 + x * std::exp(-x);
  };
  const auto sol = solve_bvp(
      [](double) { return 2.0; }, [](double) { return 1.0; },
      [](double x) { return std::exp(x); }, 0, 1, exact(0), exact(1), 200);
  EXPECT_LT(max_error(sol, exact), 1e-5);
}
