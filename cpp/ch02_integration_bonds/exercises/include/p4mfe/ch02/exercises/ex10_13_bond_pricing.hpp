#pragma once
// Exercises 10-13 (section 2.12): pricing coupon bonds off a zero rate curve
// and off an instantaneous rate curve, and the par yield.

#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch02/rates.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace p4mfe::ch02::ex {

// ----------------------------------------------------------- Exercises 10-11 ---
// The continuously compounded 6, 12, 18 and 24 month zero rates are 5%,
// 5.25%, 5.35% and 5.5%.
// 10. Find the price of a two year semiannual coupon bond with coupon rate 5%.
// 11. What is the par yield for a 2-year semiannual coupon bond?
//
// Discount factors D_i = e^(-r(0,t_i) t_i):
//   D(0.5) = e^(-0.025)   = 0.975310
//   D(1)   = e^(-0.0525)  = 0.948854
//   D(1.5) = e^(-0.08025) = 0.922886
//   D(2)   = e^(-0.11)    = 0.895834
// 10. Face 100: coupons 2.5 at 0.5, 1, 1.5, 2 and 100 at 2:
//   B = 2.5 (D1 + D2 + D3 + D4) + 100 D4 = 98.9406.
// 11. The par yield is the coupon rate c that makes the bond trade at par:
//   100 = (100 c/2) sum_i D_i + 100 D4   =>   c = 2 (1 - D4) / sum_i D_i = 5.5661%.
inline constexpr std::array<double, 4> ex10_times = {0.5, 1.0, 1.5, 2.0};
inline constexpr std::array<double, 4> ex10_zero_rates = {0.05, 0.0525, 0.0535, 0.055};

[[nodiscard]] inline std::array<double, 4> ex10_discount_factors() {
    std::array<double, 4> d{};
    for (std::size_t i = 0; i < d.size(); ++i) d[i] = discount_factor(ex10_zero_rates[i], ex10_times[i]);
    return d;
}

[[nodiscard]] inline double ex10_price(double coupon_rate = 0.05) {
    const auto d = ex10_discount_factors();
    // The bond's cash flow dates are exactly the curve's dates.
    return bond_price_from_discount(bond_cash_flows(2.0, coupon_rate, 2), [&](double t) {
        for (std::size_t i = 0; i < d.size(); ++i)
            if (std::abs(t - ex10_times[i]) < 1e-12) return d[i];
        throw std::invalid_argument("ex10: no zero rate for this date");
    });
}

[[nodiscard]] inline double ex11_par_yield() {
    const auto d = ex10_discount_factors();
    double annuity = 0.0;
    for (double v : d) annuity += v;
    return 2.0 * (1.0 - d.back()) / annuity;
}

// ----------------------------------------------------------- Exercise 12 ---
// The instantaneous rate curve is r(t) = 0.05 + 0.005 ln(1 + t), t >= 0.
// (i)   Find the corresponding zero rate curve.
// (ii)  Compute the 6, 12, 18 and 24 month discount factors.
// (iii) Find the price of a two year semiannual coupon bond with coupon rate 5%.
//
// (i) integral_0^t ln(1 + s) ds = (1 + t) ln(1 + t) - t, so
//   r(0, t) = (1/t) integral_0^t r(s) ds = 0.05 + 0.005 ((1 + t) ln(1 + t) - t) / t,
// with r(0, 0) = r(0) = 0.05 in the limit.
// (ii) D(0, t) = exp(-0.05 t - 0.005 ((1 + t) ln(1 + t) - t)):
//   D(0.5) = 0.974782,  D(1) = 0.949394,  D(1.5) = 0.924083,  D(2) = 0.898994.
// (iii) B = 2.5 (D(0.5) + D(1) + D(1.5) + D(2)) + 100 D(2) = 99.2675.
[[nodiscard]] inline double ex12_inst_rate(double t) { return 0.05 + 0.005 * std::log1p(t); }
[[nodiscard]] inline double ex12_integral(double t) { return 0.05 * t + 0.005 * ((1.0 + t) * std::log1p(t) - t); }
[[nodiscard]] inline double ex12_zero_rate(double t) { return t == 0.0 ? 0.05 : ex12_integral(t) / t; }
[[nodiscard]] inline double ex12_discount_factor(double t) { return std::exp(-ex12_integral(t)); }
[[nodiscard]] inline double ex12_price(double coupon_rate = 0.05) {
    return bond_price_from_discount(bond_cash_flows(2.0, coupon_rate, 2), ex12_discount_factor);
}

// ----------------------------------------------------------- Exercise 13 ---
// The instantaneous rate curve is r(t) = 0.05 / (1 + exp(-(1 + t)^2)).
// (i)  Use Simpson's rule to compute the 1 and 2 year discount factors with six
//      decimal digits accuracy, and the 3 year discount factor with eight.
// (ii) Find the value of a three year annual coupon bond with coupon rate 5%
//      (and face value 100).
//
// (i) D(0, t) = exp(-integral_0^t r(s) ds). r has no elementary antiderivative,
// so integrate with Simpson's rule, doubling n from 4 until consecutive values
// agree to tol. Since |d/dI e^(-I)| = e^(-I) < 1, an error in the integral of
// at most tol gives an error in D of less than tol; tol = 1e-6 (resp. 1e-8)
// is enough. The driver stops at n = 8, 16 and 64:
//   D(0, 1) = 0.956595,  D(0, 2) = 0.910128,  D(0, 3) = 0.86574103.
// (ii) Coupons of 5 at t = 1, 2, 3 and 100 at t = 3:
//   B = 5 (D(0,1) + D(0,2) + D(0,3)) + 100 D(0,3) = 100.2364.
// The bond trades above par because the rates (between 4.4% and 5%) are below
// the 5% coupon.
[[nodiscard]] inline double ex13_inst_rate(double t) { return 0.05 / (1.0 + std::exp(-(1.0 + t) * (1.0 + t))); }

[[nodiscard]] inline IntegrationResult ex13_integral(double t, double tol) {
    return integrate_to_tolerance(ex13_inst_rate, 0.0, t, tol, Rule::Simpson, 4);
}
[[nodiscard]] inline double ex13_discount_factor(double t, double tol) {
    return std::exp(-ex13_integral(t, tol).value);
}
// The accuracy the exercise asks for: 1e-6 for t <= 2, 1e-8 for t = 3.
[[nodiscard]] inline double ex13_tol(double t) { return t > 2.0 ? 1e-8 : 1e-6; }

[[nodiscard]] inline double ex13_price(double coupon_rate = 0.05) {
    return bond_price_from_discount(bond_cash_flows(3.0, coupon_rate, 1),
                                    [](double t) { return ex13_discount_factor(t, ex13_tol(t)); });
}

}  // namespace p4mfe::ch02::ex
