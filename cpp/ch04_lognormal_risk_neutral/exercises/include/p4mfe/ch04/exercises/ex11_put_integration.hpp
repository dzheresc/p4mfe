#pragma once
// Exercise 11 (section 4.9): a European put as a risk-neutral integral over
// a finite interval, priced with the Midpoint and Simpson's rules and checked
// against Black-Scholes.

#include <cmath>
#include <cstddef>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch04/asset_model.hpp>
#include <vector>

namespace p4mfe::ch04::ex {

// ----------------------------------------------------------- Exercise 11 ---
// Six months plain vanilla European put with strike 50 on a lognormal asset
// paying dividends continuously at 2%; interest rates constant at 4%. Use
// risk-neutral valuation to write the value of the put as an integral over a
// finite interval. Find the value with six decimal digits accuracy using the
// Midpoint Rule and Simpson's Rule, compute the Black-Scholes value P_BS and
// report the approximation errors of the numerical integration at each step.
//
// Assumption: the text does not give the spot price or the volatility. We
// take S_0 = 50 (at the money) and sigma = 0.3, the volatility of Exercises 7
// and 9; every function takes the market as a parameter.
//
// Under the risk-neutral measure S_T is lognormal with density
//   f(x) = 1/(x sigma sqrt(2 pi T)) exp(-(ln(x/S_0) - (r - q - sigma^2/2) T)^2
//   / (2 sigma^2 T)), x > 0.
// The payoff (K - x)^+ vanishes for x >= K, so
//   P = e^(-rT) E_RN[(K - S_T)^+] = e^(-rT) Integral_0^K (K - x) f(x) dx,
// an integral over the finite interval [0, K]. The integrand is smooth on
// [0, K]: f and all its derivatives tend to 0 as x -> 0, so f(0) = 0 is used
// by Simpson's rule. (Only the payoff kink at x = K is not smooth, and it is
// an endpoint.)
//
// Following the book's tolerance algorithm (section 2.4), start with n = 4
// intervals and double n until two successive approximations differ by less
// than 5e-7. With the market above,
//   P_BS = 50 e^(-0.02) N(-d2) - 50 e^(-0.01) N(-d1) = 3.918999,
// and the run (see the printout) stops at n = 8192 for the Midpoint rule
// (value 3.918999402, error 5.7e-8) and at n = 128 for Simpson's rule (value
// 3.918999352, error 6.9e-9). Simpson's errors fall by about 16 per doubling
// and the Midpoint errors by about 4: orders 4 and 2. (The Midpoint value is
// already within 5e-7 of P_BS at n = 2048, but the stopping test compares
// successive approximations, which is all one can do without P_BS.)
struct Ex11Market {
  double spot0 = 50.0, strike = 50.0, maturity = 0.5, sigma = 0.3, r = 0.04,
         q = 0.02;
};
// e^(-rT) (K - x) f(x), the integrand on [0, K].
[[nodiscard]] inline double Ex11Integrand(double x, const Ex11Market& m = {}) {
  return std::exp(-m.r * m.maturity) * (m.strike - x) *
         PricePdf(x, m.spot0, m.maturity, m.sigma, m.r, m.q);
}
[[nodiscard]] inline double Ex11PutBs(const Ex11Market& m = {}) {
  return ch03::BsPut(m.spot0, m.strike, m.maturity, m.sigma, m.r, m.q);
}
[[nodiscard]] inline double Ex11PutNumeric(std::size_t n, ch02::Rule rule,
                                           const Ex11Market& m = {}) {
  return ch02::Integrate([&](double x) { return Ex11Integrand(x, m); }, 0.0,
                         m.strike, n, rule);
}
struct Ex11Step {
  std::size_t n;
  double value;
  double error;  // value - P_BS
};
// Every approximation of the run n = n0, 2 n0, ..., up to the first n with
// |I_n - I_(n/2)| < tol; the last entry is the accepted value.
[[nodiscard]] inline std::vector<Ex11Step> Ex11Run(ch02::Rule rule,
                                                   double tol = 5e-7,
                                                   std::size_t n0 = 4,
                                                   const Ex11Market& m = {}) {
  const double exact = Ex11PutBs(m);
  std::vector<Ex11Step> steps;
  for (std::size_t n = n0; n <= (std::size_t{1} << 24); n *= 2) {
    const double v = Ex11PutNumeric(n, rule, m);
    steps.push_back({n, v, v - exact});
    if (steps.size() > 1 && std::abs(v - steps[steps.size() - 2].value) < tol)
      break;
  }
  return steps;
}

}  // namespace p4mfe::ch04::ex
