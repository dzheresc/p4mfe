#include <gtest/gtest.h>

#include <cmath>
#include <p4mfe/ch07/finite_differences.hpp>

using namespace p4mfe::ch07;

namespace {
const auto kF = [](double x) { return std::exp(-x * x / 2) * std::sin(x); };
// f'(x) and f''(x) by hand.
double Df(double x) {
  return std::exp(-x * x / 2) * (std::cos(x) - x * std::sin(x));
}
double D2f(double x) {
  return std::exp(-x * x / 2) *
         ((x * x - 2) * std::sin(x) - 2 * x * std::cos(x));
}
}  // namespace

// ---------------------------------------------------------------- 7.1 -----
TEST(Ch07FiniteDifferences, Schemes) {
  const double a = 0.7, h = 1e-3;
  EXPECT_NEAR(ForwardDifference(kF, a, h), Df(a), 1e-3);
  EXPECT_NEAR(BackwardDifference(kF, a, h), Df(a), 1e-3);
  EXPECT_NEAR(CentralDifference(kF, a, h), Df(a), 1e-6);
  EXPECT_NEAR(SecondCentralDifference(kF, a, h), D2f(a), 1e-5);
  // Forward and backward errors are +-h/2 f''(a) to leading order, so their
  // average is the central difference.
  EXPECT_NEAR(ForwardDifference(kF, a, h) - Df(a), 0.5 * h * D2f(a), 1e-6);
  EXPECT_NEAR(BackwardDifference(kF, a, h) - Df(a), -0.5 * h * D2f(a), 1e-6);
  EXPECT_NEAR(
      0.5 * (ForwardDifference(kF, a, h) + BackwardDifference(kF, a, h)),
      CentralDifference(kF, a, h), 1e-12);
  EXPECT_EQ(FirstDifference(kF, a, h, FdScheme::kBackward),
            BackwardDifference(kF, a, h));
  EXPECT_THROW((void)CentralDifference(kF, a, 0.0), std::invalid_argument);
}

TEST(Ch07FiniteDifferences, ExactForLowDegreePolynomials) {
  // One-sided differences are exact for linear functions, central ones for
  // quadratics (first derivative) and cubics (second derivative).
  const auto lin = [](double x) { return 3 * x - 1; };
  const auto quad = [](double x) { return x * x - 3 * x; };
  const auto cubic = [](double x) { return x * x * x; };
  EXPECT_NEAR(ForwardDifference(lin, 2.0, 0.5), 3.0, 1e-12);
  EXPECT_NEAR(CentralDifference(quad, 2.0, 0.5), 1.0, 1e-12);
  EXPECT_NEAR(SecondCentralDifference(cubic, 2.0, 0.5), 12.0, 1e-12);
}

TEST(Ch07FiniteDifferences, ObservedOrder) {
  const double a = 0.3;
  EXPECT_EQ(TruncationOrder(FdScheme::kForward), 1);
  EXPECT_EQ(TruncationOrder(FdScheme::kCentral), 2);
  EXPECT_NEAR(FdObservedOrder(kF, a, Df(a), FdScheme::kForward, 1e-2), 1.0,
              0.05);
  EXPECT_NEAR(FdObservedOrder(kF, a, Df(a), FdScheme::kBackward, 1e-2), 1.0,
              0.05);
  EXPECT_NEAR(FdObservedOrder(kF, a, Df(a), FdScheme::kCentral, 1e-2), 2.0,
              0.05);
  EXPECT_NEAR(FdObservedOrder(kF, a, D2f(a), FdScheme::kCentral, 1e-2, 2), 2.0,
              0.05);
}

// -------------------------------------------------------------- 7.3.1 -----
TEST(Ch07FiniteDifferences, RoundOffAndOptimalStep) {
  const double a = 1.0;
  const auto rows = FdErrorTable(kF, a, Df(a), FdScheme::kCentral, 0.1, 12);
  ASSERT_EQ(rows.size(), 12u);
  EXPECT_DOUBLE_EQ(rows[3].h, 1e-4);
  // Error falls ~100x per row while truncation dominates...
  EXPECT_NEAR(rows[0].error / rows[1].error, 100.0, 5.0);
  // ...then round-off wins: h = 1e-11 is far worse than h = 1e-5.
  EXPECT_GT(std::abs(rows[10].error), 100 * std::abs(rows[4].error));
  // The optimal steps.
  EXPECT_NEAR(OptimalStep(FdScheme::kForward), 1.49e-8, 1e-10);
  EXPECT_NEAR(OptimalStep(FdScheme::kCentral), 6.06e-6, 1e-8);
  EXPECT_NEAR(OptimalStep(FdScheme::kCentral, 2), 1.22e-4, 1e-6);
  EXPECT_DOUBLE_EQ(OptimalStep(FdScheme::kCentral, 1, -50.0),
                   50 * OptimalStep(FdScheme::kCentral));
  // Errors at the optimal steps are near the best achievable.
  for (FdScheme s : {FdScheme::kForward, FdScheme::kCentral}) {
    const double h = OptimalStep(s, 1, a);
    EXPECT_LT(std::abs(FirstDifference(kF, a, h, s) - Df(a)),
              s == FdScheme::kCentral ? 1e-10 : 1e-7);
  }
  EXPECT_LT(std::abs(SecondCentralDifference(
                         kF, a, OptimalStep(FdScheme::kCentral, 2, a)) -
                     D2f(a)),
            1e-7);
  // Second differences are more fragile: at h = 1e-7 round-off is ~eps/h^2.
  EXPECT_GT(std::abs(SecondCentralDifference(kF, a, 1e-7) - D2f(a)),
            1e3 * std::abs(CentralDifference(kF, a, 1e-7) - Df(a)));
}
