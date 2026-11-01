#include <p4mfe/ch02/integration.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <stdexcept>

using namespace p4mfe::ch02;

namespace {
const auto gauss = [](double x) { return std::exp(-x * x); };
// integral_0^2 e^(-x^2) dx = sqrt(pi)/2 erf(2), the book's 2.4.2 example.
const double gauss_0_2 = 0.5 * std::sqrt(std::numbers::pi) * std::erf(2.0);
}  // namespace

TEST(Ch02Rules, ExactForLowDegreePolynomials) {
    // Midpoint and trapezoidal are exact for lines, Simpson for cubics.
    const auto line = [](double x) { return 3 * x - 1; };
    const auto cubic = [](double x) { return x * x * x - 2 * x * x + x + 5; };
    EXPECT_NEAR(midpoint_rule(line, 0, 2, 1), 4.0, 1e-14);
    EXPECT_NEAR(trapezoidal_rule(line, 0, 2, 1), 4.0, 1e-14);
    // integral_0^2 cubic = 4 - 16/3 + 2 + 10 = 32/3
    EXPECT_NEAR(simpson_rule(cubic, 0, 2, 1), 32.0 / 3.0, 1e-13);
}

TEST(Ch02Rules, SimpsonIsWeightedTrapezoidalAndMidpoint) {
    for (std::size_t n : {1u, 4u, 17u})
        EXPECT_NEAR(simpson_rule(gauss, 0, 2, n),
                    (trapezoidal_rule(gauss, 0, 2, n) + 2 * midpoint_rule(gauss, 0, 2, n)) / 3, 1e-14);
}

TEST(Ch02Rules, MidpointAndTrapezoidalBracketConvexIntegrand) {
    // For convex f, midpoint underestimates and trapezoidal overestimates.
    const auto f = [](double x) { return std::exp(x); };
    const double exact = std::exp(1.0) - 1.0;
    EXPECT_LT(midpoint_rule(f, 0, 1, 8), exact);
    EXPECT_GT(trapezoidal_rule(f, 0, 1, 8), exact);
    // ... and the midpoint error is about half the trapezoidal one.
    EXPECT_NEAR((exact - midpoint_rule(f, 0, 1, 64)) / (trapezoidal_rule(f, 0, 1, 64) - exact), 0.5, 1e-3);
}

TEST(Ch02Convergence, ObservedOrders) {
    EXPECT_NEAR(observed_order(gauss, 0, 2, gauss_0_2, 32, Rule::Midpoint), 2.0, 0.05);
    EXPECT_NEAR(observed_order(gauss, 0, 2, gauss_0_2, 32, Rule::Trapezoidal), 2.0, 0.05);
    EXPECT_NEAR(observed_order(gauss, 0, 2, gauss_0_2, 16, Rule::Simpson), 4.0, 0.1);
}

TEST(Ch02Convergence, ConcreteExampleToTolerance) {
    // 2.4.2: integral_0^2 e^(-x^2) dx to tolerance 1e-7 with each rule.
    for (Rule rule : {Rule::Midpoint, Rule::Trapezoidal, Rule::Simpson}) {
        const auto res = integrate_to_tolerance(gauss, 0, 2, 1e-7, rule);
        EXPECT_NEAR(res.value, gauss_0_2, 1e-7);
    }
    // Simpson needs far fewer intervals than the second-order rules.
    EXPECT_LT(integrate_to_tolerance(gauss, 0, 2, 1e-7, Rule::Simpson).intervals,
              integrate_to_tolerance(gauss, 0, 2, 1e-7, Rule::Midpoint).intervals / 8);
}

TEST(Ch02Improper, InfiniteLimits) {
    EXPECT_NEAR(improper_integral([](double x) { return 1 / (x * x); }, 1, infinity).value, 1.0, 1e-12);
    EXPECT_NEAR(improper_integral([](double x) { return std::exp(-x); }, 0, infinity).value, 1.0, 1e-8);
    EXPECT_NEAR(improper_integral([](double x) { return std::exp(-x * x / 2); }, -infinity, infinity).value,
                std::sqrt(2 * std::numbers::pi), 1e-8);
    EXPECT_NEAR(improper_integral([](double x) { return std::exp(x); }, -infinity, 0).value, 1.0, 1e-8);
    EXPECT_NEAR(improper_integral([](double x) { return 1 / (1 + x * x); }, -infinity, infinity).value,
                std::numbers::pi, 1e-8);
    EXPECT_THROW((void)improper_integral(gauss, 1, 0), std::invalid_argument);
}

TEST(Ch02Improper, GammaFunction) {
    EXPECT_NEAR(gamma_function(1), 1.0, 1e-8);
    EXPECT_NEAR(gamma_function(5), 24.0, 1e-7);
    EXPECT_NEAR(gamma_function(2.5), std::tgamma(2.5), 1e-8);
    // Gamma(x + 1) = x Gamma(x)
    EXPECT_NEAR(gamma_function(4.3), 3.3 * gamma_function(3.3), 1e-7);
}

TEST(Ch02DifferentiateImproper, UnderTheIntegralSign) {
    // d/dt integral_0^inf e^(-t x) dx = d/dt (1/t) = -1/t^2
    const double t = 1.7;
    EXPECT_NEAR(improper_integral_derivative([](double x, double s) { return std::exp(-s * x); }, 0, infinity, t),
                -1 / (t * t), 1e-7);
    // F(t) = integral_0^inf e^(-x^2) cos(t x) dx = sqrt(pi)/2 e^(-t^2/4)
    const double expected = -t / 2 * std::sqrt(std::numbers::pi) / 2 * std::exp(-t * t / 4);
    EXPECT_NEAR(improper_integral_derivative([](double x, double s) { return std::exp(-x * x) * std::cos(s * x); },
                                             0, infinity, t),
                expected, 1e-7);
}
