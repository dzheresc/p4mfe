#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch08/exercises/ex09_10_heat_change_of_variables.hpp>
#include <p4mfe/ch08/exercises/ex11_barrier_rebate.hpp>
#include <p4mfe/ch08/exercises/ex12_13_asian_pdes.hpp>
#include <p4mfe/ch08/heat_equation.hpp>

using namespace p4mfe;
using namespace p4mfe::ch08::ex;
using ch03::OptionType;

TEST(Ch08Ex09, InitialConditions) {
  const double strike = 50.0, sigma = 0.3, r = 0.05, q = 0.02;
  const auto c = Ex09ComputeCoefficients(sigma, r, q);
  for (double x = -1.0; x <= 1.0; x += 0.1) {
    EXPECT_NEAR(Ex09U(OptionType::kCall, x, 0.0, strike, sigma, r, q),
                Ex09CallU0(x, strike, c.a), 1e-12);
    EXPECT_NEAR(Ex09U(OptionType::kPut, x, 0.0, strike, sigma, r, q),
                Ex09PutU0(x, strike, c.a), 1e-12);
  }
  // The library's transform: a' = -a, b' = -b.
  const auto lib = ch08::MakeHeatTransform(sigma, r, q);
  EXPECT_NEAR(lib.a, -c.a, 1e-14);
  EXPECT_NEAR(lib.b, -c.b, 1e-12);
}

TEST(Ch08Ex09, HeatEquation) {
  const double strike = 50.0;
  for (double q : {0.0, 0.03})
    for (OptionType type : {OptionType::kCall, OptionType::kPut})
      for (double x : {-0.3, 0.0, 0.2})
        for (double tau : {0.01, 0.05}) {
          // u_xx is of order K / sqrt(tau) here; the residual is the
          // O(h^2) central difference error, h = 1e-4.
          EXPECT_NEAR(Ex09HeatResidual(type, x, tau, strike, 0.3, 0.05, q), 0.0,
                      1e-2)
              << x << " " << tau;
        }
}

TEST(Ch08Ex10, SolutionOfTheSystem) {
  for (double q : {0.0, 0.01, 0.04})
    for (double sigma : {0.2, 0.4}) {
      const double r = 0.05;
      const auto s = Ex10Solve(sigma, r, q);
      const auto res = Ex10ComputeResiduals(s.a, s.b, sigma, r, q);
      EXPECT_NEAR(res.first, 0.0, 1e-13);
      EXPECT_NEAR(res.second, 0.0, 1e-12);
      const auto c = Ex09ComputeCoefficients(sigma, r, q);
      EXPECT_NEAR(s.a, c.a, 1e-14);
      EXPECT_NEAR(s.b, c.b, 1e-12);
      // The printed b solves the system only for q = 0.
      const auto printed =
          Ex10ComputeResiduals(s.a, Ex10PrintedB(sigma, r, q), sigma, r, q);
      EXPECT_NEAR(printed.second, -2.0 * q / (sigma * sigma), 1e-12);
    }
}

TEST(Ch08Ex11, BoundaryCondition) {
  const double spot = 40.0, strike = 40.0, barrier = 35.0, maturity = 0.5,
               sigma = 0.3, r = 0.05, rebate = 2.0;
  const auto c = Ex09ComputeCoefficients(sigma, r);
  // u at the barrier corresponds to V = R.
  for (double tau : {0.0, 0.01, 0.0225}) {
    const double u = Ex11UBoundary(tau, barrier, strike, rebate, sigma, r);
    EXPECT_NEAR(std::exp(-c.a * std::log(barrier / strike) - c.b * tau) * u,
                rebate, 1e-13);
  }
  EXPECT_DOUBLE_EQ(Ex11RebateValue(barrier, barrier, maturity, sigma, r), 1.0);
  // Heat equation with the derived boundary condition vs. closed form.
  for (double q : {0.0, 0.02})
    for (double paid : {0.0, rebate}) {
      const double cf = Ex11DownAndOutCallWithRebate(
          spot, strike, barrier, paid, maturity, sigma, r, q);
      EXPECT_NEAR(
          Ex11HeatFd(spot, strike, barrier, paid, maturity, sigma, r, q, 400),
          cf, 2e-4)
          << q << " " << paid;
    }
  EXPECT_NEAR(Ex11DownAndOutCallWithRebate(spot, strike, barrier, 0.0, maturity,
                                           sigma, r),
              3.398883, 1e-6);
  EXPECT_NEAR(Ex11DownAndOutCallWithRebate(spot, strike, barrier, rebate,
                                           maturity, sigma, r),
              4.439140, 1e-6);
}

TEST(Ch08Ex12, GeometricAsianPdes) {
  const AsianMarket m;
  const auto geo_price = [&](double y, double t) {
    return Ex12GeometricAsianCall(y, t, m);
  };
  for (double y : {std::log(45.0), std::log(50.0), std::log(56.0)})
    for (double t : {0.2, 0.6})
      EXPECT_NEAR(Ex12FResidual(geo_price, y, t, m.maturity, m.sigma, m.r), 0.0,
                  1e-5)
          << y << " " << t;
  const auto value = [&](double spot, double integral, double t) {
    return geo_price(Ex12Y(spot, integral, t, m.maturity), t);
  };
  for (double spot : {44.0, 50.0, 57.0}) {
    const double t = 0.3, integral = t * std::log(48.0);
    EXPECT_NEAR(Ex12VResidual(value, spot, integral, t, m.sigma, m.r, 1e-5),
                0.0, 2e-5)
        << spot;
  }
  // At t = 0, I = 0 the price is the classic geometric Asian call:
  // Black-Scholes with volatility sigma/sqrt(3) and yield (r + sigma^2/6)/2.
  const double spot = 50.0;
  const double bs =
      ch03::BsCall(spot, m.strike, m.maturity, m.sigma / std::sqrt(3.0), m.r,
                   0.5 * (m.r + m.sigma * m.sigma / 6.0));
  EXPECT_NEAR(value(spot, 0.0, 0.0), bs, 1e-12);
}

TEST(Ch08Ex12, OperatorIdentity) {
  // For any smooth F, residual_(8.52)(F(y(S, I, t), t)) = residual_F(F).
  const double maturity = 1.0, sigma = 0.25, r = 0.03;
  const auto smooth_f = [](double y, double t) {
    return std::sin(y) * std::exp(t) + y * y * y / 10.0;
  };
  const auto value = [&](double spot, double integral, double t) {
    return smooth_f(Ex12Y(spot, integral, t, maturity), t);
  };
  for (double spot : {30.0, 60.0})
    for (double t : {0.25, 0.7}) {
      const double integral = 0.4;
      EXPECT_NEAR(Ex12VResidual(value, spot, integral, t, sigma, r),
                  Ex12FResidual(smooth_f, Ex12Y(spot, integral, t, maturity), t,
                                maturity, sigma, r),
                  1e-5);
    }
}

TEST(Ch08Ex13, ArithmeticAsianPdes) {
  const double maturity = 1.0, sigma = 0.3, r = 0.05;
  const auto linear_h = [&](double ratio, double t) {
    return Ex13HLinear(ratio, t, maturity, r);
  };
  const auto value = [&](double spot, double integral, double t) {
    return spot * linear_h(integral / spot, t);
  };
  for (double ratio : {0.0, 0.3, 1.2})
    EXPECT_NEAR(Ex13HResidual(linear_h, ratio, 0.4, sigma, r), 0.0, 1e-7);
  for (double spot : {40.0, 55.0})
    EXPECT_NEAR(Ex13VResidual(value, spot, 12.0, 0.4, sigma, r), 0.0,
                1e-5 * spot);
  // Generic smooth H: residual_(8.53)(S H(I/S, t)) = S * residual_(8.55)(H).
  const auto generic_h = [](double ratio, double t) {
    return std::exp(-ratio * ratio) * std::cos(t) + ratio * ratio * ratio;
  };
  const auto generic_v = [&](double spot, double integral, double t) {
    return spot * generic_h(integral / spot, t);
  };
  for (double spot : {20.0, 45.0})
    for (double integral : {5.0, 30.0}) {
      const double t = 0.5;
      EXPECT_NEAR(Ex13VResidual(generic_v, spot, integral, t, sigma, r),
                  spot * Ex13HResidual(generic_h, integral / spot, t, sigma, r),
                  1e-4 * spot * (1.0 + std::pow(integral / spot, 3)))
          << spot << " " << integral;
    }
}
