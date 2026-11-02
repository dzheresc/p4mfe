#pragma once
// Exercise 16 (section 6.11): bond price after a yield change, with and
// without the convexity adjustment.

#include <array>
#include <cstddef>
#include <p4mfe/ch02/bonds.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 16 ---
// A five year bond worth 101 has duration 1.5 years and convexity 2.5. Use
// (2.57), without convexity, and (6.73), with it, to find the price of the
// bond if the yield goes up by 10bp, 50bp, 1% and 2%.
//
//   (2.57)  B_new ~ B (1 - D dy),
//   (6.73)  B_new ~ B (1 - D dy + C dy^2 / 2).
// With B = 101, D = 1.5, C = 2.5:
//   dy      (2.57)       (6.73)       convexity term B C dy^2 / 2
//   0.001   100.848500   100.848626   0.000126
//   0.005   100.242500   100.245656   0.003156
//   0.01     99.485000    99.497625   0.012625
//   0.02     97.970000    98.020500   0.050500
// The convexity correction grows like dy^2: negligible at 10bp, 5 cents per
// 101 at 2%. (No cash flows are given, so the exact prices cannot be
// computed; a duration of 1.5 for a 5 year bond means most of its value is
// paid early, e.g. a large coupon or an amortizing bond.)
struct Ex16Row {
  double dy, duration_only, duration_convexity;
};
inline constexpr std::array<double, 4> kEx16YieldChanges{0.001, 0.005, 0.01,
                                                         0.02};

[[nodiscard]] inline std::array<Ex16Row, 4> Ex16Table(double price = 101.0,
                                                      double duration = 1.5,
                                                      double convexity = 2.5) {
  std::array<Ex16Row, 4> out{};
  for (std::size_t i = 0; i < out.size(); ++i) {
    const double dy = kEx16YieldChanges[i];
    out[i] = {
        dy, ch02::PriceApproxDuration(price, duration, dy),
        ch02::PriceApproxDurationConvexity(price, duration, convexity, dy)};
  }
  return out;
}

}  // namespace p4mfe::ch06::ex
