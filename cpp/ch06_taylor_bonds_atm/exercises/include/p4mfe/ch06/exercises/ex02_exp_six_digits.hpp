#pragma once
// Exercise 2 (section 6.11): e^0.25 to six decimal digits from the Taylor
// series of e^x.

#include <cmath>
#include <p4mfe/ch06/taylor_series.hpp>

namespace p4mfe::ch06::ex {

// ------------------------------------------------------------ Exercise 2 ---
// Use the Taylor series expansion of e^x to find the value of e^0.25 with six
// decimal digits accuracy.
//
// P_n(x) = sum_{k=0}^n x^k / k!. By the Lagrange form of the remainder, for
// x > 0 there is c in (0, x) with
//   e^x - P_n(x) = e^c x^(n+1) / (n+1)!  <  e^x x^(n+1) / (n+1)!  <  3^x
//   x^(n+1) / (n+1)!
// (e < 3, so the bound does not use the value we are computing). "Six decimal
// digits" means an error below 0.5e-6. For x = 0.25 and 3^0.25 = 1.316074:
//   n = 4:  1.316074 * 0.25^5 / 5! = 1.07e-5    too big,
//   n = 5:  1.316074 * 0.25^6 / 6! = 4.46e-7    < 5e-7.
// So six terms suffice:
//   P_5(0.25) = 1 + 0.25 + 0.03125 + 0.0026041667 + 0.0001627604 + 0.0000081380
//             = 1.2840250651,
// i.e. e^0.25 = 1.284025 to six decimals (the exact value is 1.2840254167;
// the actual error of P_5 is 3.52e-7).
struct Ex02Result {
  unsigned n;    // degree of the Taylor polynomial used
  double value;  // P_n(x)
  double bound;  // 3^x x^(n+1) / (n+1)!
};

[[nodiscard]] inline double Ex02ErrorBound(double x, unsigned n) {
  return std::pow(3.0, x) * std::pow(x, n + 1) /
         std::tgamma(static_cast<double>(n) + 2.0);
}

// Smallest n whose remainder bound is below tol (x >= 0).
[[nodiscard]] inline Ex02Result Ex02Solve(double x = 0.25,
                                          double tol = 0.5e-6) {
  unsigned n = 0;
  while (Ex02ErrorBound(x, n) >= tol) ++n;
  return {n, ExpSeries(x, n), Ex02ErrorBound(x, n)};
}

}  // namespace p4mfe::ch06::ex
