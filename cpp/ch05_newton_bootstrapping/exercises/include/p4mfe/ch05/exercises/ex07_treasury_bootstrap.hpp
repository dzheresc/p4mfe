#pragma once
// Exercise 7 (section 5.7): bootstrapping a zero rate curve from Treasury
// prices, with continuously and with semiannually compounded zero rates.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch05/bootstrapping.hpp>
#include <p4mfe/ch05/solvers.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch05::ex {

// ------------------------------------------------------------ Exercise 7 ---
// (i) Bootstrap a zero rate curve from these Treasury prices (semiannual
//     coupons), with continuously compounded interest:
//       3-month T-bill    0        98.7
//       6-month T-bill    0        97.5
//       2-year T-bond     4.875    100 5/32
//       3-year T-bond     4.875    100 5/32
//       5-year T-bond     4.625    99 22/32
//       10-year T-bond    4.875    101 4/32
// (ii) How do the zero rate curves bootstrapped with semiannually compounded
//      and with continuously compounded interest compare? Will one be higher
//      or lower than the other, and why?
//
// (i) Section 5.5: the curve is linear between bootstrapped maturities and
// flat before 3 months (no overnight rate is given; no cash flow falls before
// 3 months anyway). The bills give r(0, 0.25) and r(0, 0.5) directly from
// 100 e^(-r T) = price. Each bond then adds one point, found with Newton's
// method, its coupons between the last known maturity and its own maturity
// being priced with linearly interpolated rates (bootstrap_zero_curve):
//   t      0.25      0.5       2         3         5         10
//   r    5.2341%   5.0636%   4.7289%   4.7582%   4.6303%   4.6772%
//
// (ii) The bond prices fix the discount factors. A rate compounded twice a
// year r_2 and a continuously compounded rate r give the same discount
// factor when (1 + r_2/2)^(-2t) = e^(-r t), i.e.
//   r_2 = 2 (e^(r/2) - 1) > r   (e^x > 1 + x for x != 0),
// so the semiannual curve is higher, by about r^2/4 (~6 bp at 5%). The
// interpolation assumption is "linear in the quoted rate", which is not the
// same assumption for the two conventions, so the semiannual curve is not
// exactly the converted continuous one away from the bills; the difference
// is about 1e-4 bp here (zero at the bills, where no interpolation is
// used), while the gap between the curves is 5.4-6.9 bp. Bootstrapped
// semiannually:
//   t      0.25      0.5       2         3         5         10
//   r_2  5.3032%   5.1282%   4.7853%   4.8152%   4.6844%   4.7323%

struct Ex07Quotes {
  std::vector<ch02::BondQuote> quotes{{0.25, 0.0, 98.7},
                                      {0.5, 0.0, 97.5},
                                      {2.0, 0.04875, 100.0 + 5.0 / 32.0},
                                      {3.0, 0.04875, 100.0 + 5.0 / 32.0},
                                      {5.0, 0.04625, 99.0 + 22.0 / 32.0},
                                      {10.0, 0.04875, 101.0 + 4.0 / 32.0}};
};

// (i)
[[nodiscard]] inline ch02::ZeroCurve ex07_curve_continuous(
    const Ex07Quotes& q = {}) {
  return bootstrap_zero_curve(q.quotes, 2);
}

// Discount factor for a rate compounded m times a year, m = 0 meaning
// continuously, and its derivative with respect to the rate.
[[nodiscard]] inline double ex07_discount(double r, double t, int m) {
  return m == 0 ? std::exp(-r * t) : std::pow(1.0 + r / m, -m * t);
}
[[nodiscard]] inline double ex07_discount_dr(double r, double t, int m) {
  return m == 0 ? -t * std::exp(-r * t)
                : -t * std::pow(1.0 + r / m, -m * t - 1.0);
}

// Section 5.5's bootstrapping with zero rates compounded m times a year
// (m = 0: continuously, which reproduces bootstrap_zero_curve). Linear
// interpolation of the m-compounded rate between maturities, flat before
// the first one.
[[nodiscard]] inline ch02::ZeroCurve ex07_bootstrap_compounded(
    std::vector<ch02::BondQuote> quotes, int m, int freq = 2) {
  std::sort(quotes.begin(), quotes.end(),
            [](const ch02::BondQuote& a, const ch02::BondQuote& b) {
              return a.maturity < b.maturity;
            });
  ch02::ZeroCurve curve;
  for (const auto& bq : quotes) {
    const auto cf = ch02::bond_cash_flows(bq.maturity, bq.coupon_rate, freq);
    const double T = bq.maturity;
    const bool empty = curve.times.empty();
    const double T_last = empty ? 0.0 : curve.times.back();
    const double r_last = empty ? 0.0 : curve.rates.back();
    double pv_known = 0.0;
    std::vector<double> c, t, w;
    for (std::size_t i = 0; i < cf.times.size(); ++i) {
      if (!empty && cf.times[i] <= T_last) {
        pv_known +=
            cf.amounts[i] *
            ex07_discount(zero_rate_linear(curve, cf.times[i]), cf.times[i], m);
      } else {
        c.push_back(cf.amounts[i]);
        t.push_back(cf.times[i]);
        w.push_back(empty ? 1.0 : (cf.times[i] - T_last) / (T - T_last));
      }
    }
    const auto rate = [&](double x, std::size_t i) {
      return r_last + (x - r_last) * w[i];
    };
    const auto f = [&](double x) {
      double B = pv_known;
      for (std::size_t i = 0; i < c.size(); ++i)
        B += c[i] * ex07_discount(rate(x, i), t[i], m);
      return B - bq.price;
    };
    const auto df = [&](double x) {
      double d = 0.0;
      for (std::size_t i = 0; i < c.size(); ++i)
        d += c[i] * w[i] * ex07_discount_dr(rate(x, i), t[i], m);
      return d;
    };
    curve.times.push_back(T);
    curve.rates.push_back(
        newton(f, df, empty ? 0.05 : r_last, {1e-12, 1e-12, 100}).x);
  }
  return curve;
}

// (ii)
[[nodiscard]] inline ch02::ZeroCurve ex07_curve_semiannual(
    const Ex07Quotes& q = {}) {
  return ex07_bootstrap_compounded(q.quotes, 2);
}
// r_m equivalent to a continuously compounded r: m (e^(r/m) - 1).
[[nodiscard]] inline double ex07_to_compounded(double r, int m) {
  return m * (std::exp(r / m) - 1.0);
}

}  // namespace p4mfe::ch05::ex
