#pragma once
// Shared helper for the option-strategy exercises (18-29): a static portfolio
// of cash, shares and European options with a common maturity T, its payoff
// at T as a function of S(T), its cost at time 0, and a payoff table that
// stands in for the book's payoff diagrams.

#include <cstddef>
#include <p4mfe/ch01/options.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p4mfe::ch01::ex {

// One option position. quantity > 0 is long, < 0 is short.
struct OptionLeg {
  OptionType type;
  double strike;
  double quantity;
  double price = 0.0;  // premium per option at time 0 (used by cost())
};

struct Portfolio {
  double cash = 0.0;  // zero-coupon cash paying `cash` at T (rates are zero in
                      // these exercises)
  double shares = 0.0;  // units of the underlying asset
  std::vector<OptionLeg> options;

  // V(T) as a function of S(T).
  [[nodiscard]] double Payoff(double spot_t) const noexcept {
    double v = cash + shares * spot_t;
    for (const auto& leg : options)
      v += leg.quantity * ::p4mfe::ch01::Payoff(leg.type, spot_t, leg.strike);
    return v;
  }

  // Cost at time 0 of the option legs, the shares at spot S0 and the cash
  // (zero rates: one unit paid at T costs one unit today).
  [[nodiscard]] double Cost(double spot0 = 0.0) const noexcept {
    double c = cash + shares * spot0;
    for (const auto& leg : options) c += leg.quantity * leg.price;
    return c;
  }

  // P&L at maturity with zero interest rates: payoff minus cost.
  [[nodiscard]] double Pnl(double spot_t, double spot0 = 0.0) const noexcept {
    return Payoff(spot_t) - Cost(spot0);
  }
};

// Rows (S(T), V(T)) on n + 1 evenly spaced points in [lo, hi]; print or plot
// them to get the payoff diagram.
[[nodiscard]] inline std::vector<std::pair<double, double>> PayoffTable(
    const Portfolio& p, double lo, double hi, std::size_t n) {
  if (n == 0 || hi <= lo)
    throw std::invalid_argument("payoff_table: need n > 0 and hi > lo");
  std::vector<std::pair<double, double>> rows;
  rows.reserve(n + 1);
  const double h = (hi - lo) / static_cast<double>(n);
  for (std::size_t i = 0; i <= n; ++i) {
    const double s = lo + static_cast<double>(i) * h;
    rows.emplace_back(s, p.Payoff(s));
  }
  return rows;
}

// Convenience constructors.
[[nodiscard]] inline OptionLeg Call(double strike, double quantity,
                                    double price = 0.0) {
  return {OptionType::kCall, strike, quantity, price};
}
[[nodiscard]] inline OptionLeg Put(double strike, double quantity,
                                   double price = 0.0) {
  return {OptionType::kPut, strike, quantity, price};
}

}  // namespace p4mfe::ch01::ex
