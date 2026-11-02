#pragma once
// Chapter 2, sections 2.6-2.10: bonds and their yield, duration and
// convexity, discretely compounded yields, zero coupon bonds, and the
// numerical side of bond math (pricing off a rate curve, Newton's method for
// the yield, bootstrapping zero rates).
//
// A bond is a list of cash flows c_i paid at times t_i (years from today).
// Its price for a continuously compounded yield y is
//   B(y) = sum_i c_i e^(-y t_i).

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rates.hpp"

namespace p4mfe::ch02 {

struct CashFlows {
  std::vector<double> times;    // t_i, increasing
  std::vector<double> amounts;  // c_i
};

// ---------------------------------------------------------------- 2.6 ------
// Cash flows of a bond with face value `face`, annual coupon rate
// `coupon_rate` paid `freq` times a year, maturing at T. Coupon dates are
// T, T - 1/freq, T - 2/freq, ... down to the first one after today, so the
// bond may be between coupon dates. Each coupon is face * coupon_rate / freq;
// the last cash flow also repays the face value.
[[nodiscard]] inline CashFlows BondCashFlows(double maturity,
                                             double coupon_rate, int freq,
                                             double face = 100.0) {
  if (maturity <= 0.0)
    throw std::invalid_argument("bond_cash_flows: maturity must be positive");
  if (freq <= 0)
    throw std::invalid_argument("bond_cash_flows: freq must be positive");
  const double dt = 1.0 / freq;
  const double coupon = face * coupon_rate / freq;
  CashFlows cf;
  if (coupon != 0.0) {
    // Number of coupon dates in (0, T]; the small slack absorbs rounding in
    // T*freq.
    const auto k = static_cast<std::size_t>(std::ceil(maturity * freq - 1e-9));
    cf.times.reserve(k);
    cf.amounts.reserve(k);
    for (std::size_t i = k; i >= 1; --i) {
      cf.times.push_back(maturity - static_cast<double>(i - 1) * dt);
      cf.amounts.push_back(coupon);
    }
    cf.amounts.back() += face;
  } else {
    cf.times.push_back(maturity);
    cf.amounts.push_back(face);
  }
  return cf;
}

namespace detail {
inline void Check(const CashFlows& cf) {
  if (cf.times.size() != cf.amounts.size() || cf.times.empty())
    throw std::invalid_argument(
        "cash flows: times and amounts must be non-empty and the same size");
}
}  // namespace detail

// B = sum_i c_i e^(-y t_i).
[[nodiscard]] inline double BondPrice(const CashFlows& cf, double y) {
  detail::Check(cf);
  double price = 0.0;
  for (std::size_t i = 0; i < cf.times.size(); ++i)
    price += cf.amounts[i] * std::exp(-y * cf.times[i]);
  return price;
}

// B = sum_i c_i D(0, t_i) for any discount function D(t).
template <std::invocable<double> D>
[[nodiscard]] double BondPriceFromDiscount(const CashFlows& cf, D&& disc) {
  detail::Check(cf);
  double price = 0.0;
  for (std::size_t i = 0; i < cf.times.size(); ++i)
    price += cf.amounts[i] * disc(cf.times[i]);
  return price;
}

// B = sum_i c_i e^(-r(0, t_i) t_i) for a zero rate curve r(0, t).
template <std::invocable<double> Z>
[[nodiscard]] double BondPriceFromZeroRate(const CashFlows& cf, Z&& zero) {
  return BondPriceFromDiscount(
      cf, [&](double t) { return std::exp(-zero(t) * t); });
}

// B = sum_i c_i exp(-integral_0^{t_i} r(s) ds) for an instantaneous rate
// curve r(s) (2.10). Integrates each [t_{i-1}, t_i] once and accumulates, so
// the cash flow times must be non-negative and non-decreasing.
template <std::invocable<double> R>
[[nodiscard]] double BondPriceFromInstRate(const CashFlows& cf, R&& r,
                                           double tol = 1e-12) {
  detail::Check(cf);
  double price = 0.0;
  double integral = 0.0;
  double t_prev = 0.0;
  for (std::size_t i = 0; i < cf.times.size(); ++i) {
    if (cf.times[i] < t_prev)
      throw std::invalid_argument(
          "bond_price_from_inst_rate: cash flow times must be non-negative and "
          "increasing");
    if (cf.times[i] > t_prev)
      integral += IntegrateToTolerance(r, t_prev, cf.times[i], tol).value;
    t_prev = cf.times[i];
    price += cf.amounts[i] * std::exp(-integral);
  }
  return price;
}

// ---------------------------------------------------------------- 2.7 ------
// Duration and convexity with respect to the yield:
//   D = -(1/B) dB/dy  = (1/B) sum_i t_i   c_i e^(-y t_i)
//   C =  (1/B) d2B/dy2 = (1/B) sum_i t_i^2 c_i e^(-y t_i)
struct BondMetrics {
  double price;
  double duration;
  double convexity;
};

[[nodiscard]] inline BondMetrics ComputeBondMetrics(const CashFlows& cf,
                                                    double y) {
  detail::Check(cf);
  double price = 0.0, price_t = 0.0, price_tt = 0.0;
  for (std::size_t i = 0; i < cf.times.size(); ++i) {
    const double t = cf.times[i];
    const double v = cf.amounts[i] * std::exp(-y * t);
    price += v;
    price_t += t * v;
    price_tt += t * t * v;
  }
  return {price, price_t / price, price_tt / price};
}

[[nodiscard]] inline double BondDuration(const CashFlows& cf, double y) {
  return ComputeBondMetrics(cf, y).duration;
}
[[nodiscard]] inline double BondConvexity(const CashFlows& cf, double y) {
  return ComputeBondMetrics(cf, y).convexity;
}

// Dollar duration B*D = -dB/dy, and DV01 = B*D / 10000, the price change for
// a one basis point drop in yield.
[[nodiscard]] inline double DollarDuration(const CashFlows& cf, double y) {
  const auto m = ComputeBondMetrics(cf, y);
  return m.price * m.duration;
}
[[nodiscard]] inline double Dv01(const CashFlows& cf, double y) {
  return DollarDuration(cf, y) / 10000.0;
}

// Price after a yield change dy, from the Taylor expansion of B(y):
//   B(y + dy) ~ B (1 - D dy)                     (duration only)
//   B(y + dy) ~ B (1 - D dy + C dy^2 / 2)        (duration and convexity)
[[nodiscard]] inline double PriceApproxDuration(double price, double duration,
                                                double dy) noexcept {
  return price * (1.0 - duration * dy);
}
[[nodiscard]] inline double PriceApproxDurationConvexity(double price,
                                                         double duration,
                                                         double convexity,
                                                         double dy) noexcept {
  return price * (1.0 - duration * dy + 0.5 * convexity * dy * dy);
}

// ---------------------------------------------------------------- 2.8 ------
// Yield y compounded m times a year:
//   B = sum_i c_i (1 + y/m)^(-m t_i)
//   Macaulay duration  D_mac = (1/B) sum_i t_i c_i (1 + y/m)^(-m t_i)
//   Modified duration  D_mod = -(1/B) dB/dy = D_mac / (1 + y/m)
[[nodiscard]] inline double BondPriceDiscrete(const CashFlows& cf, double y,
                                              int m) {
  detail::Check(cf);
  double price = 0.0;
  for (std::size_t i = 0; i < cf.times.size(); ++i)
    price += cf.amounts[i] * DiscountFactorDiscrete(y, cf.times[i], m);
  return price;
}

[[nodiscard]] inline double MacaulayDuration(const CashFlows& cf, double y,
                                             int m) {
  detail::Check(cf);
  double price = 0.0, price_t = 0.0;
  for (std::size_t i = 0; i < cf.times.size(); ++i) {
    const double v = cf.amounts[i] * DiscountFactorDiscrete(y, cf.times[i], m);
    price += v;
    price_t += cf.times[i] * v;
  }
  return price_t / price;
}

[[nodiscard]] inline double ModifiedDuration(const CashFlows& cf, double y,
                                             int m) {
  return MacaulayDuration(cf, y, m) / (1.0 + y / m);
}

// ---------------------------------------------------------------- 2.9 ------
// A zero coupon bond pays `face` at T only. Its price is face * D(0, T), its
// yield is the zero rate r(0, T), and its duration is T.
[[nodiscard]] inline double ZeroCouponBondPrice(double maturity, double r,
                                                double face = 100.0) noexcept {
  return face * std::exp(-r * maturity);
}
[[nodiscard]] inline double ZeroRateFromZeroCouponPrice(double price,
                                                        double maturity,
                                                        double face = 100.0) {
  return ZeroRateFromDiscount(price / face, maturity);
}

// A coupon bond is a portfolio of zero coupon bonds, one per cash flow, so
// bond prices pin down zero rates. Bootstrapping: sort bonds by maturity; each
// bond's cash flows before maturity must fall on earlier maturities, whose
// discount factors are known, which leaves one unknown D(0, T):
//   D(0, T) = (B - sum_{t_i < T} c_i D(0, t_i)) / c_T.
struct BondQuote {
  double maturity;
  double coupon_rate;
  double price;
};

struct ZeroCurve {
  std::vector<double> times;
  std::vector<double> rates;  // continuously compounded zero rates r(0, t)
};

[[nodiscard]] inline ZeroCurve BootstrapZeroRates(std::vector<BondQuote> quotes,
                                                  int freq,
                                                  double face = 100.0) {
  std::sort(quotes.begin(), quotes.end(),
            [](const BondQuote& a, const BondQuote& b) {
              return a.maturity < b.maturity;
            });
  // Quotes are sorted by maturity, so `times` grows in increasing order and
  // each bond's coupon dates can be matched with one forward scan instead
  // of a linear search per date (O(N^2) overall instead of O(N^3)).
  constexpr double kTimeTolerance = 1e-9;
  std::vector<double> times, discounts;
  times.reserve(quotes.size());
  discounts.reserve(quotes.size());
  for (const auto& q : quotes) {
    const auto cf = BondCashFlows(q.maturity, q.coupon_rate, freq, face);
    double pv = 0.0;
    std::size_t known = 0;  // first earlier maturity not before the coupon date
    for (std::size_t i = 0; i + 1 < cf.times.size(); ++i) {
      const double t = cf.times[i];
      while (known < times.size() && times[known] < t - kTimeTolerance) ++known;
      if (known == times.size() || std::abs(times[known] - t) >= kTimeTolerance)
        throw std::invalid_argument(
            "bootstrap_zero_rates: a coupon date has no earlier bond");
      pv += cf.amounts[i] * discounts[known];
    }
    if (!times.empty() && std::abs(times.back() - q.maturity) < kTimeTolerance)
      throw std::invalid_argument("bootstrap_zero_rates: duplicate maturity");
    times.push_back(q.maturity);
    discounts.push_back((q.price - pv) / cf.amounts.back());
  }
  ZeroCurve curve;
  curve.rates.reserve(times.size());
  for (std::size_t j = 0; j < times.size(); ++j)
    curve.rates.push_back(ZeroRateFromDiscount(discounts[j], times[j]));
  curve.times = std::move(times);
  return curve;
}

// --------------------------------------------------------------- 2.10 ------
// Yield of a bond with price B by Newton's method on f(y) = B(y) - B:
//   y_{k+1} = y_k - f(y_k) / f'(y_k)
//           = y_k + (sum_i c_i e^(-y_k t_i) - B) / (sum_i t_i c_i e^(-y_k
//           t_i)).
// B(y) is decreasing and convex, so from y0 = 0.1 the iteration converges
// quickly for any realistic price. Stops when |y_{k+1} - y_k| <= tol.
[[nodiscard]] inline double BondYield(const CashFlows& cf, double price,
                                      double tol = 1e-12, double y0 = 0.1,
                                      int max_iter = 100) {
  detail::Check(cf);
  double y = y0;
  for (int k = 0; k < max_iter; ++k) {
    double model_price = 0.0, model_price_t = 0.0;
    for (std::size_t i = 0; i < cf.times.size(); ++i) {
      const double v = cf.amounts[i] * std::exp(-y * cf.times[i]);
      model_price += v;
      model_price_t += cf.times[i] * v;
    }
    const double step = (model_price - price) / model_price_t;
    y += step;
    if (std::abs(step) <= tol) return y;
  }
  throw std::runtime_error("bond_yield: Newton's method did not converge");
}

}  // namespace p4mfe::ch02
