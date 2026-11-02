#pragma once
// Chapter 5, section 5.5: bootstrapping for finding zero rate curves.
//
// Chapter 2 (bootstrap_zero_rates) bootstraps only when every coupon date of
// each bond is the maturity of an earlier bond, so each new bond adds one
// linear equation in one discount factor. Real bond sets are sparser: a 5-year
// bond pays coupons at 3.5, 4 and 4.5 years even if the only other bond
// matures at 3 years.
//
// The book's fix: assume the zero rate curve r(0, t) is linear between the
// maturities where it is known. With the curve known up to T_last and a new
// bond maturing at T > T_last, every rate in (T_last, T] is a linear function
// of the one unknown x = r(0, T),
//   r(0, t) = r_last + (x - r_last) (t - T_last) / (T - T_last),
// and pricing the bond gives one nonlinear equation in x,
//   f(x) = sum_i c_i e^(-r(0, t_i) t_i) - B = 0,
//   f'(x) = -sum_{t_i > T_last} c_i t_i w_i e^(-r(0, t_i) t_i),
//   w_i = (t_i - T_last) / (T - T_last),
// solved with Newton's method (5.1.2). Bonds are processed by maturity.
//
// The starting curve may hold known short rates, e.g. the overnight rate as
// r(0, 0). Before its first point the curve is extended flat; with no
// starting curve at all, the first bond is priced at a flat rate x (its
// yield).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch05/solvers.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch05 {

using ch02::BondQuote;
using ch02::CashFlows;
using ch02::ZeroCurve;

// r(0, t) by linear interpolation between the curve's points, flat outside
// them. Times must be increasing.
[[nodiscard]] inline double zero_rate_linear(const ZeroCurve& curve, double t) {
  const auto& ts = curve.times;
  const auto& rs = curve.rates;
  if (ts.empty() || ts.size() != rs.size())
    throw std::invalid_argument("zero_rate_linear: bad curve");
  if (t <= ts.front()) return rs.front();
  if (t >= ts.back()) return rs.back();
  const auto hi = static_cast<std::size_t>(
      std::upper_bound(ts.begin(), ts.end(), t) - ts.begin());
  const std::size_t lo = hi - 1;
  const double w = (t - ts[lo]) / (ts[hi] - ts[lo]);
  return rs[lo] + w * (rs[hi] - rs[lo]);
}

// One bootstrapping step: the zero rate r(0, T) at the maturity T of a bond
// with cash flows `cf` and price `price`, given the curve up to T_last < T.
[[nodiscard]] inline RootResult bootstrap_next_rate(
    const ZeroCurve& curve, const CashFlows& cf, double price,
    const Tolerances& tol = {1e-12, 1e-12, 100}) {
  ch02::detail::Check(cf);
  const double T = cf.times.back();
  const bool empty = curve.times.empty();
  const double T_last = empty ? 0.0 : curve.times.back();
  const double r_last = empty ? 0.0 : curve.rates.back();
  if (!empty && !(T > T_last))
    throw std::invalid_argument(
        "bootstrap_next_rate: maturity must exceed the curve");

  // Split the cash flows: rates known from the curve, or interpolated with
  // weight w.
  double pv_known = 0.0;
  std::vector<double> c, t, w;
  for (std::size_t i = 0; i < cf.times.size(); ++i) {
    const double ti = cf.times[i];
    if (!empty && ti <= T_last) {
      pv_known += cf.amounts[i] * std::exp(-zero_rate_linear(curve, ti) * ti);
    } else {
      c.push_back(cf.amounts[i]);
      t.push_back(ti);
      w.push_back(empty ? 1.0 : (ti - T_last) / (T - T_last));
    }
  }
  const auto rate = [&](double x, std::size_t i) {
    return r_last + (x - r_last) * w[i];
  };
  const auto f = [&](double x) {
    double B = pv_known;
    for (std::size_t i = 0; i < c.size(); ++i)
      B += c[i] * std::exp(-rate(x, i) * t[i]);
    return B - price;
  };
  const auto df = [&](double x) {
    double d = 0.0;
    for (std::size_t i = 0; i < c.size(); ++i)
      d -= c[i] * t[i] * w[i] * std::exp(-rate(x, i) * t[i]);
    return d;
  };
  return newton(f, df, empty ? 0.05 : r_last, tol);
}

// Bootstraps the whole curve: starts from `curve` (possibly empty) and adds
// one point (T, r(0, T)) per bond, in order of maturity.
[[nodiscard]] inline ZeroCurve bootstrap_zero_curve(
    std::vector<BondQuote> quotes, int freq, ZeroCurve curve = {},
    double face = 100.0, const Tolerances& tol = {1e-12, 1e-12, 100}) {
  std::sort(quotes.begin(), quotes.end(),
            [](const BondQuote& a, const BondQuote& b) {
              return a.maturity < b.maturity;
            });
  for (const auto& q : quotes) {
    const auto cf = ch02::BondCashFlows(q.maturity, q.coupon_rate, freq, face);
    const double r = bootstrap_next_rate(curve, cf, q.price, tol).x;
    curve.times.push_back(q.maturity);
    curve.rates.push_back(r);
  }
  return curve;
}

}  // namespace p4mfe::ch05
