#pragma once
// Standard normal density and cumulative distribution, used throughout the
// book (Black-Scholes, Greeks, implied volatility, ...).

#include <cmath>
#include <limits>
#include <numbers>

namespace p4mfe {

// Standard normal density: phi(x) = exp(-x^2/2) / sqrt(2*pi).
[[nodiscard]] inline double norm_pdf(double x) noexcept {
    return std::exp(-0.5 * x * x) * std::numbers::inv_sqrtpi / std::numbers::sqrt2;
}

// Standard normal cumulative distribution N(x), via the complementary error
// function so that the left tail keeps full relative precision.
[[nodiscard]] inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x / std::numbers::sqrt2);
}

// Inverse of N: the x with N(x) = p, for 0 < p < 1 (returns -inf / +inf at
// p = 0 / 1). Acklam's rational approximation (relative error ~1e-9) refined
// by one Halley step on N(x) - p, which brings it to full double precision.
// Used for quantiles and confidence intervals of normal and lognormal
// variables (chapter 4).
[[nodiscard]] inline double norm_inv(double p) noexcept {
    if (!(p > 0.0)) return p == 0.0 ? -std::numeric_limits<double>::infinity()
                                    : std::numeric_limits<double>::quiet_NaN();
    if (!(p < 1.0)) return p == 1.0 ? std::numeric_limits<double>::infinity()
                                    : std::numeric_limits<double>::quiet_NaN();
    constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                            1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
    constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                            6.680131188771972e+01, -1.328068155288572e+01};
    constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                            -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
    constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                            3.754408661907416e+00};
    constexpr double p_low = 0.02425;

    double x;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - p_low) {
        const double q = p - 0.5;
        const double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log1p(-p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    // Halley step: f = N(x) - p, f' = phi(x), f'' = -x phi(x).
    const double e = norm_cdf(x) - p;
    const double u = e / norm_pdf(x);
    return x - u / (1.0 + 0.5 * x * u);
}

}  // namespace p4mfe
