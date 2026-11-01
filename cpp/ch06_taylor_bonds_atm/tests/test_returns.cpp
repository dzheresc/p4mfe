#include <p4mfe/ch06/returns.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace p4mfe::ch06;

// ------------------------------------------------------------------ 6.4 ---
TEST(Ch06Returns, PercentageAndLogReturns) {
    EXPECT_DOUBLE_EQ(percentage_return(50, 55), 0.1);
    EXPECT_DOUBLE_EQ(log_return(50, 55), std::log(1.1));
    EXPECT_DOUBLE_EQ(percentage_return(50, 54, 1), 0.1);
    EXPECT_DOUBLE_EQ(log_return(50, 54, 1), std::log(1.1));
    for (double R : {-0.3, -0.01, 0.02, 0.25}) {
        EXPECT_NEAR(percentage_from_log(log_from_percentage(R)), R, 1e-16);
        EXPECT_LT(log_from_percentage(R), R);  // r < R
        // r - (R - R^2/2) is O(R^3); the cubic term R^3/3 dominates it.
        const double r = log_from_percentage(R);
        EXPECT_LE(std::abs(r - log_return_taylor(R, 2)), std::abs(R * R * R) / 3 / (1 - std::abs(R)) + 1e-17);
        EXPECT_NEAR(log_return_taylor(R, 60), r, 1e-9);
    }
    EXPECT_DOUBLE_EQ(log_return_taylor(0.1, 1), 0.1);
    EXPECT_THROW((void)percentage_return(0, 1), std::invalid_argument);
    EXPECT_THROW((void)log_return(1, -1), std::invalid_argument);
}

TEST(Ch06Returns, LogReturnsAddPercentageReturnsCompound) {
    const std::vector<double> S{100, 103, 99.5, 104, 110.2, 108};
    const auto R = percentage_returns(S);
    const auto r = log_returns(S);
    ASSERT_EQ(R.size(), 5u);
    EXPECT_NEAR(sum_log_returns(r), log_return(S.front(), S.back()), 1e-15);
    EXPECT_NEAR(compound_percentage_returns(R), percentage_return(S.front(), S.back()), 1e-15);
    // Summing percentage returns is only an approximation.
    EXPECT_GT(std::abs(std::accumulate(R.begin(), R.end(), 0.0) - percentage_return(S.front(), S.back())), 1e-4);
}

// ---------------------------------------------------------------- 6.4.1 ---
TEST(Ch06Returns, PortfolioReturns) {
    const std::vector<double> units{100, -50, 200};
    const std::vector<double> S1{40, 60, 25}, S2{42, 57, 26.5};
    const auto w = portfolio_weights(units, S1);
    EXPECT_NEAR(std::accumulate(w.begin(), w.end(), 0.0), 1.0, 1e-15);
    double V1 = 0, V2 = 0;
    std::vector<double> R, r;
    for (std::size_t i = 0; i < 3; ++i) {
        V1 += units[i] * S1[i];
        V2 += units[i] * S2[i];
        R.push_back(percentage_return(S1[i], S2[i]));
        r.push_back(log_return(S1[i], S2[i]));
    }
    EXPECT_NEAR(portfolio_percentage_return(w, R), percentage_return(V1, V2), 1e-15);
    EXPECT_NEAR(portfolio_log_return(w, r), log_return(V1, V2), 1e-15);
    const double err = std::abs(portfolio_log_return_approx(w, r) - log_return(V1, V2));
    EXPECT_GT(err, 0.0);
    EXPECT_LT(err, 5e-3);
    EXPECT_THROW((void)portfolio_weights(units, std::vector<double>{1, 2}), std::invalid_argument);
}
