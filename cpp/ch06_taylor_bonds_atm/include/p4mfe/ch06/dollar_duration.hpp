#pragma once
// Chapter 6, sections 6.7-6.7.2: dollar duration and dollar convexity, DV01,
// and bond portfolio immunization.
//
// For any position whose value V depends on the yield (or on a parallel
// shift of the yield curve) y,
//   dollar duration   D$ = -dV/dy  (= V D when V != 0),
//   dollar convexity  C$ =  d2V/dy2 (= V C when V != 0),
// and by Taylor's formula
//   dV ~ -D$ dy + C$ dy^2 / 2.
// Unlike D and C, D$ and C$ are additive across positions, and they stay
// meaningful for portfolios worth zero (e.g. a long/short bond position or a
// hedged portfolio), where D = -V'/V is undefined.
//
// DV01 (6.7.1), the "dollar value of a basis point", is the value gained when
// the yield drops by 1bp = 0.0001:
//   DV01 = -(1/10000) dV/dy = D$ / 10000 ~ V(y - 0.0001) - V(y).
//
// Immunization (6.7.2): a portfolio with value V, dollar duration D$ and
// dollar convexity C$ is hedged with q1, q2 units of two bonds and cash so
// that the hedged position has zero value, zero D$ and zero C$:
//   q1 D$_1 + q2 D$_2 = -D$
//   q1 C$_1 + q2 C$_2 = -C$
//   cash = -(V + q1 B_1 + q2 B_2).
// The value then changes only by O(dy^3) under a small parallel shift.
// With one hedging bond only D$ can be matched (duration hedging).

#include <p4mfe/ch02/bonds.hpp>

#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace p4mfe::ch06 {

using ch02::CashFlows;

// Value and its first two (dollar) sensitivities to the yield.
struct DollarRisk {
    double value;
    double dollar_duration;   // -dV/dy
    double dollar_convexity;  //  d2V/dy2

    [[nodiscard]] constexpr double dv01() const noexcept { return dollar_duration / 10000.0; }
};

// ---------------------------------------------------------------- 6.7 ------
// Dollar convexity of one bond, B C (chapter 2 gives dollar_duration = B D).
[[nodiscard]] inline double dollar_convexity(const CashFlows& cf, double y) {
    const auto m = ch02::bond_metrics(cf, y);
    return m.price * m.convexity;
}

[[nodiscard]] inline DollarRisk bond_dollar_risk(const CashFlows& cf, double y) {
    const auto m = ch02::bond_metrics(cf, y);
    return {m.price, m.price * m.duration, m.price * m.convexity};
}

// `units` of a position with risk r.
[[nodiscard]] constexpr DollarRisk scale(const DollarRisk& r, double units) noexcept {
    return {units * r.value, units * r.dollar_duration, units * r.dollar_convexity};
}

// Sum of positions: everything is additive.
[[nodiscard]] inline DollarRisk portfolio_dollar_risk(std::span<const double> units,
                                                      std::span<const DollarRisk> risks) {
    if (units.size() != risks.size()) throw std::invalid_argument("portfolio_dollar_risk: size mismatch");
    DollarRisk p{0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < units.size(); ++i) {
        const auto s = scale(risks[i], units[i]);
        p.value += s.value;
        p.dollar_duration += s.dollar_duration;
        p.dollar_convexity += s.dollar_convexity;
    }
    return p;
}

// dV ~ -D$ dy  and  dV ~ -D$ dy + C$ dy^2 / 2.
[[nodiscard]] constexpr double value_change_approx(const DollarRisk& r, double dy) noexcept {
    return -r.dollar_duration * dy;
}
[[nodiscard]] constexpr double value_change_approx_convexity(const DollarRisk& r, double dy) noexcept {
    return -r.dollar_duration * dy + 0.5 * r.dollar_convexity * dy * dy;
}

// D = D$ / V and C = C$ / V, defined only for V != 0.
[[nodiscard]] inline double duration_from_dollar(const DollarRisk& r) {
    if (r.value == 0.0) throw std::domain_error("duration_from_dollar: zero value");
    return r.dollar_duration / r.value;
}
[[nodiscard]] inline double convexity_from_dollar(const DollarRisk& r) {
    if (r.value == 0.0) throw std::domain_error("convexity_from_dollar: zero value");
    return r.dollar_convexity / r.value;
}

// -------------------------------------------------------------- 6.7.1 ------
// DV01 from repricing: V(y - 1bp) - V(y), and the central version
// (V(y - 1bp) - V(y + 1bp)) / 2, which matches D$/10000 to O(bp^3).
[[nodiscard]] inline double dv01_reprice(const CashFlows& cf, double y) {
    return ch02::bond_price(cf, y - 1e-4) - ch02::bond_price(cf, y);
}
[[nodiscard]] inline double dv01_central(const CashFlows& cf, double y) {
    return 0.5 * (ch02::bond_price(cf, y - 1e-4) - ch02::bond_price(cf, y + 1e-4));
}

// -------------------------------------------------------------- 6.7.2 ------
// Duration hedge with a single bond: units with q D$_h = -D$.
[[nodiscard]] inline double duration_hedge_units(const DollarRisk& portfolio, const DollarRisk& hedge) {
    if (hedge.dollar_duration == 0.0) throw std::invalid_argument("duration_hedge_units: hedge has zero D$");
    return -portfolio.dollar_duration / hedge.dollar_duration;
}

struct Immunization {
    double units1;  // q1, units of the first hedging bond
    double units2;  // q2, units of the second hedging bond
    double cash;    // cash position making the total value zero
};

// Solves the 2x2 system above by Cramer's rule.
[[nodiscard]] inline Immunization immunize(const DollarRisk& portfolio, const DollarRisk& bond1,
                                           const DollarRisk& bond2) {
    const double det = bond1.dollar_duration * bond2.dollar_convexity - bond2.dollar_duration * bond1.dollar_convexity;
    const double scale_ = std::abs(bond1.dollar_duration * bond2.dollar_convexity) +
                          std::abs(bond2.dollar_duration * bond1.dollar_convexity);
    if (std::abs(det) <= 1e-12 * scale_)
        throw std::invalid_argument("immunize: hedging bonds must have different D$/C$ ratios");
    const double q1 =
        (-portfolio.dollar_duration * bond2.dollar_convexity + bond2.dollar_duration * portfolio.dollar_convexity) / det;
    const double q2 =
        (-bond1.dollar_duration * portfolio.dollar_convexity + portfolio.dollar_duration * bond1.dollar_convexity) / det;
    return {q1, q2, -(portfolio.value + q1 * bond1.value + q2 * bond2.value)};
}

}  // namespace p4mfe::ch06
