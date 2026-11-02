#pragma once
// Exercise 14 (section 8.12): in-the-money European puts worth less than
// their intrinsic value.

#include <algorithm>
#include <array>
#include <p4mfe/ch03/black_scholes.hpp>

namespace p4mfe::ch08::ex {

// ----------------------------------------------------------- Exercise 14 ---
// Non-dividend-paying asset, lognormal, S = 40, sigma = 30%, r = 5%. Fill in
// the Black-Scholes values of the ITM puts below, and say for which of them
// the intrinsic value max(K - S, 0) is larger than the Black-Scholes value (so
// the American put is guaranteed to be worth more than the European put).
//
//   Strike  Maturity   Black-Scholes   Intrinsic   Intrinsic > BS?
//     45    6 months      5.819577         5          no
//     45    3 months      5.340299         5          no
//     48    6 months      8.032528         8          no
//     48    3 months      7.823400         8          yes
//     51    6 months     10.486171        11          yes
//     51    3 months     10.547645        11          yes
//
// Why: P = K e^(-rT) N(-d2) - S N(-d1). Deep in the money N(-d1), N(-d2) -> 1
// and P -> K e^(-rT) - S < K - S: the time value of the strike, K (1 -
// e^(-rT)), outweighs the optionality. This happens for the deeper (K = 51) and
// shorter (T = 3 months, less optionality) options. An American put is worth at
// least K - S (exercise now), so in those cases it is strictly more valuable
// than the European put, and early exercise can be optimal (section 8.10).
struct Ex14Row {
  double strike;
  double maturity;
  double value;
  double intrinsic;
  bool intrinsic_larger;
};

inline constexpr double kEx14S = 40.0, kEx14Sigma = 0.3, kEx14R = 0.05;

[[nodiscard]] inline std::array<Ex14Row, 6> Ex14Table() {
  std::array<Ex14Row, 6> rows{};
  const double strikes[] = {45.0, 45.0, 48.0, 48.0, 51.0, 51.0};
  const double maturities[] = {0.5, 0.25, 0.5, 0.25, 0.5, 0.25};
  for (std::size_t i = 0; i < 6; ++i) {
    const double strike = strikes[i], maturity = maturities[i];
    const double v = ch03::BsPut(kEx14S, strike, maturity, kEx14Sigma, kEx14R);
    const double in = std::max(strike - kEx14S, 0.0);
    rows[i] = {strike, maturity, v, in, in > v};
  }
  return rows;
}

}  // namespace p4mfe::ch08::ex
