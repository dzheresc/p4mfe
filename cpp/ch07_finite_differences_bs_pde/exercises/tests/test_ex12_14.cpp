#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch07/exercises/ex12_gamma_theta.hpp>
#include <p4mfe/ch07/exercises/ex13_14_bs_pde_solutions.hpp>

using namespace p4mfe;
using namespace p4mfe::ch07;
using namespace p4mfe::ch07::ex;

TEST(Ch07Ex12, ExactWhenRatesAreZero) {
  for (double spot : {30.0, 50.0, 80.0})
    for (double tau : {0.1, 1.0, 3.0}) {
      const Ex12Market m{spot, 52.0, tau, 0.3, 0.0};
      EXPECT_NEAR(Ex12Lhs(m), 0.0, 1e-12) << spot << " " << tau;
      EXPECT_NEAR(Ex12Lhs(m, ch03::OptionType::kPut), 0.0, 1e-12)
          << spot << " " << tau;
    }
}

TEST(Ch07Ex12, FormulaWhenRateIsNonzero) {
  for (double r : {-0.01, 0.02, 0.05, 0.2})
    for (double spot : {30.0, 50.0, 80.0}) {
      const Ex12Market m{spot, 52.0, 0.75, 0.3, r};
      EXPECT_NEAR(Ex12Lhs(m), Ex12Rhs(m), 1e-12) << r << " " << spot;
      if (r > 0) {
        EXPECT_GT(Ex12Lhs(m), 0.0);
        EXPECT_LT(Ex12Lhs(m), 1.0);
      }
    }
  // -> 0 as r -> 0
  EXPECT_LT(std::abs(Ex12Lhs({50.0, 52.0, 0.75, 0.3, 1e-6})), 1e-4);
}

TEST(Ch07Ex13, CallSolvesThePde) {
  const PdeMarket m;
  for (double spot : {20.0, 40.0, 60.0})
    for (double t : {0.0, 0.5, 0.9}) {
      EXPECT_NEAR(Ex13ResidualGreeks(spot, t, m), 0.0, 1e-12)
          << spot << " " << t;
      EXPECT_NEAR(Ex13ResidualFd(spot, t, m), 0.0, 1e-5) << spot << " " << t;
    }
  // A non-solution for contrast: the call price with the wrong volatility in
  // the PDE.
  EXPECT_GT(std::abs(BsPdeResidualFd(
                [&](double s, double u) { return Ex13Call(s, u, m); }, 40.0,
                0.5, 0.4, m.r, m.q)),
            0.1);
}

TEST(Ch07Ex14, ForwardSolvesThePde) {
  for (double sigma : {0.1, 0.3, 0.8}) {
    const PdeMarket m{40.0, 1.0, sigma, 0.05, 0.02};
    for (double spot : {20.0, 40.0, 60.0})
      for (double t : {0.0, 0.5, 0.9}) {
        EXPECT_NEAR(Ex14ResidualExact(spot, t, m), 0.0, 1e-12)
            << spot << " " << t;
        EXPECT_NEAR(Ex14ResidualFd(spot, t, m), 0.0, 1e-6) << spot << " " << t;
      }
  }
  // f = C - P (put-call parity), so the put solves the PDE too.
  const PdeMarket m;
  const double spot = 45.0, t = 0.3;
  const double put =
      ch03::BsPut(spot, m.strike, m.maturity - t, m.sigma, m.r, m.q);
  EXPECT_NEAR(Ex13Call(spot, t, m) - put, Ex14Forward(spot, t, m), 1e-12);
}
