#include <p4mfe/ch05/solvers.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace p4mfe::ch05;

namespace {
// Order of convergence from the last three iterates still well above rounding error.
double observed_order(const std::vector<double>& xs, double root) {
    std::vector<double> e;
    for (double x : xs)
        if (std::abs(x - root) > 1e-11) e.push_back(std::abs(x - root));
    const auto n = e.size();
    return convergence_order(e[n - 3], e[n - 2], e[n - 1]);
}
}  // namespace

TEST(Ch05Solvers, AllThreeFindSqrt2) {
    const auto f = [](double x) { return x * x - 2.0; };
    const auto df = [](double x) { return 2.0 * x; };
    const Tolerances tol{1e-12, 1e-12};
    const auto b = bisection(f, 1.0, 2.0, tol);
    const auto n = newton(f, df, 1.0, tol);
    const auto s = secant(f, 2.0, 1.0, tol);
    for (double x : {b.x, n.x, s.x}) EXPECT_NEAR(x, std::sqrt(2.0), 1e-12);
    EXPECT_LT(n.iterations, s.iterations);
    EXPECT_LT(s.iterations, b.iterations);
    EXPECT_EQ(n.iterates.size(), n.iterations + 1);  // x_0 included
    EXPECT_EQ(b.iterates.size(), b.iterations);      // midpoints only
}

TEST(Ch05Solvers, BisectionHalvesTheBracket) {
    // Interval tolerance only: needs ceil(log2((b - a) / tol_int)) + 1 midpoints.
    const auto res = bisection([](double x) { return std::cos(x) - x; }, 0.0, 1.0, {1.0, 1e-6});
    EXPECT_EQ(res.iterations, 21u);
    EXPECT_NEAR(res.x, 0.7390851332151607, 1e-6);
}

TEST(Ch05Solvers, BothStoppingCriteriaMustHold) {
    // f is very flat near its root 1: |f| <= tol_approx is met long before x is accurate.
    const auto f = [](double x) { return 1e-6 * (x - 1.0); };
    const auto res = newton(f, [](double) { return 1e-6; }, 3.0, {1e-3, 1e-12});
    EXPECT_NEAR(res.x, 1.0, 1e-12);
    const auto b = bisection(f, 0.0, 3.0, {1e-3, 1e-9});
    EXPECT_NEAR(b.x, 1.0, 1e-9);
}

TEST(Ch05Solvers, ConvergenceOrders) {
    // f(x) = e^x - 2 has the simple root ln 2.
    const auto f = [](double x) { return std::exp(x) - 2.0; };
    const double root = std::log(2.0);
    const Tolerances tol{1e-15, 1e-15};
    EXPECT_NEAR(observed_order(newton(f, [](double x) { return std::exp(x); }, 2.0, tol).iterates, root), 2.0, 0.1);
    EXPECT_NEAR(observed_order(secant(f, 2.1, 2.0, tol).iterates, root), 1.618, 0.15);
}

TEST(Ch05Solvers, QuarticWithLogisticTerm) {
    // x^4 - 5x^2 + 4 - 1/(1 + e^(x^3)) has a root near 0.95 (and three others);
    // all methods started near it agree.
    const auto f = [](double x) { return std::pow(x, 4) - 5 * x * x + 4 - 1 / (1 + std::exp(std::pow(x, 3))); };
    const auto df = [](double x) {
        const double e = std::exp(std::pow(x, 3));
        return 4 * std::pow(x, 3) - 10 * x + 3 * x * x * e / ((1 + e) * (1 + e));
    };
    const auto b = bisection(f, 0.5, 1.0);
    const auto n = newton(f, df, 0.5);
    const auto s = secant(f, 0.49, 0.5);
    EXPECT_NEAR(f(n.x), 0.0, 1e-12);
    EXPECT_NEAR(b.x, n.x, 1e-9);
    EXPECT_NEAR(s.x, n.x, 1e-9);
}

TEST(Ch05Solvers, Failures) {
    const auto f = [](double x) { return x * x + 1.0; };
    EXPECT_THROW((void)bisection(f, -1.0, 1.0), std::invalid_argument);
    EXPECT_THROW((void)bisection(f, 1.0, -1.0), std::invalid_argument);
    EXPECT_THROW((void)newton(f, [](double x) { return 2 * x; }, 0.0), std::runtime_error);  // f'(0) = 0
    // Newton on x^2 + 1 never converges: it wanders.
    EXPECT_THROW((void)newton(f, [](double x) { return 2 * x; }, 0.5, {1e-9, 1e-9, 50}), std::runtime_error);
}

TEST(Ch05Solvers, ExactRootStopsImmediately) {
    const auto f = [](double x) { return x - 0.5; };
    EXPECT_EQ(newton(f, [](double) { return 1.0; }, 0.5).iterations, 0u);
    EXPECT_EQ(bisection(f, 0.0, 1.0).iterations, 1u);
}
