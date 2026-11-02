#pragma once
// Workload W3 (docs/performance-testing-plan.md): a Treasury-like zero curve,
// par bond sets for bootstrapping, and a corporate bond universe.
//
// The "true" zero curve is smooth and upward sloping:
//   r(0, t) = 0.040 + 0.012 (1 - e^(-t/3)) - 0.006 t e^(-t/2),
// i.e. about 4.0% short, a small hump, 5.2% long. Bonds are priced off it, so
// bootstrapping and yield solvers have known answers.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/ch02/bonds.hpp>
#include <random>
#include <vector>

namespace p4mfe::bench {

[[nodiscard]] inline double true_zero_rate(double t) noexcept {
  return 0.040 + 0.012 * (1.0 - std::exp(-t / 3.0)) -
         0.006 * t * std::exp(-t / 2.0);
}

// Instantaneous rate of the true curve, r(t) = d/dt (t r(0, t)) = r(0, t) + t
// dr(0, t)/dt.
[[nodiscard]] inline double true_inst_rate(double t) noexcept {
  const double dz = 0.004 * std::exp(-t / 3.0) - 0.006 * std::exp(-t / 2.0) +
                    0.003 * t * std::exp(-t / 2.0);
  return true_zero_rate(t) + t * dz;
}

// Coupon rate that prices a bond at par off the true curve.
[[nodiscard]] inline double par_coupon(double T, int freq) {
  const auto annuity = ch02::bond_price_from_zero_rate(
      ch02::bond_cash_flows(T, 1.0, freq, 100.0), true_zero_rate);
  const auto zero = ch02::bond_price_from_zero_rate(
      ch02::bond_cash_flows(T, 0.0, freq, 100.0), true_zero_rate);
  // price(c) = zero + (annuity - zero) * c, with the annuity at c = 1 (100%).
  return (100.0 - zero) / (annuity - zero);
}

// Bonds maturing at every coupon date 1/freq, 2/freq, ..., years: the set
// chapter 2's exact bootstrap needs.
[[nodiscard]] inline std::vector<ch02::BondQuote> par_bond_strip(double max_T,
                                                                 int freq) {
  std::vector<ch02::BondQuote> out;
  const auto n = static_cast<std::size_t>(std::lround(max_T * freq));
  for (std::size_t i = 1; i <= n; ++i) {
    const double T = static_cast<double>(i) / freq;
    const double c =
        std::round(par_coupon(T, freq) * 8000.0) / 8000.0;  // 1/8 bp
    out.push_back({T, c,
                   ch02::bond_price_from_zero_rate(
                       ch02::bond_cash_flows(T, c, freq), true_zero_rate)});
  }
  return out;
}

// An on-the-run style set (bills and notes/bonds) for chapter 5's bootstrap
// with linear interpolation: 0.5, 1, 2, 3, 5, 7, 10, 20, 30 years.
[[nodiscard]] inline std::vector<ch02::BondQuote> benchmark_bonds(int freq) {
  std::vector<ch02::BondQuote> out;
  for (const double T : {0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0}) {
    const double c = std::round(par_coupon(T, freq) * 8000.0) / 8000.0;
    out.push_back({T, c,
                   ch02::bond_price_from_zero_rate(
                       ch02::bond_cash_flows(T, c, freq), true_zero_rate)});
  }
  return out;
}

struct Bond {
  ch02::CashFlows cf;
  int freq;
  double coupon;
  double spread;  // over the true zero curve
  double price;
};

// Corporate bond universe: maturities U(0.25, 30), coupons 0 to 8% in 1/8
// steps, frequency 1, 2 or 4, spreads U(50, 300) bp.
[[nodiscard]] inline std::vector<Bond> bond_universe(std::size_t n,
                                                     std::uint64_t seed = 42) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> maturity(0.25, 30.0),
      spread(0.005, 0.03);
  std::uniform_int_distribution<int> coupon_eighths(0, 64), freq_pick(0, 2);
  std::vector<Bond> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double T = maturity(rng);
    const double c = coupon_eighths(rng) / 800.0;
    constexpr int freqs[] = {1, 2, 4};
    const int freq = freqs[freq_pick(rng)];
    const double s = spread(rng);
    auto cf = ch02::bond_cash_flows(T, c, freq);
    const double price = ch02::bond_price_from_zero_rate(
        cf, [s](double t) { return true_zero_rate(t) + s; });
    out.push_back({std::move(cf), freq, c, s, price});
  }
  return out;
}

[[nodiscard]] inline std::size_t bond_universe_size() {
  return by_tier<std::size_t>(100, 10'000, 1'000'000);
}

}  // namespace p4mfe::bench
