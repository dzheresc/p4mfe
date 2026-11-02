#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <p4mfe/ch06/returns.hpp>
#include <stdexcept>
#include <vector>

using namespace p4mfe::ch06;

// ------------------------------------------------------------------ 6.4 ---
TEST(Ch06Returns, PercentageAndLogReturns) {
  EXPECT_DOUBLE_EQ(PercentageReturn(50, 55), 0.1);
  EXPECT_DOUBLE_EQ(LogReturn(50, 55), std::log(1.1));
  EXPECT_DOUBLE_EQ(PercentageReturn(50, 54, 1), 0.1);
  EXPECT_DOUBLE_EQ(LogReturn(50, 54, 1), std::log(1.1));
  for (double pct_return : {-0.3, -0.01, 0.02, 0.25}) {
    EXPECT_NEAR(PercentageFromLog(LogFromPercentage(pct_return)), pct_return,
                1e-16);
    EXPECT_LT(LogFromPercentage(pct_return), pct_return);  // r < R
    // r - (R - R^2/2) is O(R^3); the cubic term R^3/3 dominates it.
    const double r = LogFromPercentage(pct_return);
    EXPECT_LE(std::abs(r - LogReturnTaylor(pct_return, 2)),
              std::abs(pct_return * pct_return * pct_return) / 3 /
                      (1 - std::abs(pct_return)) +
                  1e-17);
    EXPECT_NEAR(LogReturnTaylor(pct_return, 60), r, 1e-9);
  }
  EXPECT_DOUBLE_EQ(LogReturnTaylor(0.1, 1), 0.1);
  EXPECT_THROW((void)PercentageReturn(0, 1), std::invalid_argument);
  EXPECT_THROW((void)LogReturn(1, -1), std::invalid_argument);
}

TEST(Ch06Returns, LogReturnsAddPercentageReturnsCompound) {
  const std::vector<double> spot{100, 103, 99.5, 104, 110.2, 108};
  const auto pct_return = PercentageReturns(spot);
  const auto r = LogReturns(spot);
  ASSERT_EQ(pct_return.size(), 5u);
  EXPECT_NEAR(SumLogReturns(r), LogReturn(spot.front(), spot.back()), 1e-15);
  EXPECT_NEAR(CompoundPercentageReturns(pct_return),
              PercentageReturn(spot.front(), spot.back()), 1e-15);
  // Summing percentage returns is only an approximation.
  EXPECT_GT(
      std::abs(std::accumulate(pct_return.begin(), pct_return.end(), 0.0) -
               PercentageReturn(spot.front(), spot.back())),
      1e-4);
}

// ---------------------------------------------------------------- 6.4.1 ---
TEST(Ch06Returns, PortfolioReturns) {
  const std::vector<double> units{100, -50, 200};
  const std::vector<double> spot1{40, 60, 25}, spot2{42, 57, 26.5};
  const auto w = PortfolioWeights(units, spot1);
  EXPECT_NEAR(std::accumulate(w.begin(), w.end(), 0.0), 1.0, 1e-15);
  double value1 = 0, value2 = 0;
  std::vector<double> pct_return, r;
  for (std::size_t i = 0; i < 3; ++i) {
    value1 += units[i] * spot1[i];
    value2 += units[i] * spot2[i];
    pct_return.push_back(PercentageReturn(spot1[i], spot2[i]));
    r.push_back(LogReturn(spot1[i], spot2[i]));
  }
  EXPECT_NEAR(PortfolioPercentageReturn(w, pct_return),
              PercentageReturn(value1, value2), 1e-15);
  EXPECT_NEAR(PortfolioLogReturn(w, r), LogReturn(value1, value2), 1e-15);
  const double err =
      std::abs(PortfolioLogReturnApprox(w, r) - LogReturn(value1, value2));
  EXPECT_GT(err, 0.0);
  EXPECT_LT(err, 5e-3);
  EXPECT_THROW((void)PortfolioWeights(units, std::vector<double>{1, 2}),
               std::invalid_argument);
}
