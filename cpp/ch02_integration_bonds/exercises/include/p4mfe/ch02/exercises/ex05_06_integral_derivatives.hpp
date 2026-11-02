#pragma once
// Exercises 5-6 (section 2.12): differentiating integrals whose limits
// depend on the variable. Exercise 5 is the Delta of a call; exercise 6 gives
// the risk-neutral density as the second strike derivative of the call price.

#include <cmath>
#include <concepts>
#include <numbers>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch02::ex {

// ------------------------------------------------------------ Exercise 5 ---
// Let K, T, sigma, r > 0 and
//   g(x) = (1/sqrt(2 pi)) integral_{-inf}^{b(x)} e^(-y^2/2) dy,
//   b(x) = (ln(x/K) + (r + sigma^2/2) T) / (sigma sqrt(T)).
// Compute g'(x).
//
// Solution. g(x) = N(b(x)), and the integrand does not depend on x, so by the
// fundamental theorem of calculus and the chain rule
//   g'(x) = (1/sqrt(2 pi)) e^(-b(x)^2/2) b'(x),   b'(x) = 1 / (x sigma
//   sqrt(T)), g'(x) = e^(-b(x)^2/2) / (x sigma sqrt(2 pi T)).
// (b(x) is d1 with spot x, and g(x) = N(d1) is the Delta of a call; g'(x) is
// its Gamma, section 3.5.)
struct Ex05Params {
  double strike;
  double maturity;
  double sigma;
  double r;
};

[[nodiscard]] inline double Ex05B(const Ex05Params& p, double x) {
  return (std::log(x / p.strike) +
          (p.r + 0.5 * p.sigma * p.sigma) * p.maturity) /
         (p.sigma * std::sqrt(p.maturity));
}
[[nodiscard]] inline double Ex05G(const Ex05Params& p, double x) {
  return norm_cdf(Ex05B(p, x));
}
[[nodiscard]] inline double Ex05GPrime(const Ex05Params& p, double x) {
  const double b = Ex05B(p, x);
  return std::exp(-0.5 * b * b) /
         (x * p.sigma * std::sqrt(2.0 * std::numbers::pi * p.maturity));
}

// ------------------------------------------------------------ Exercise 6 ---
// Let h be continuous with integral_{-inf}^{inf} |x h(x)| dx finite, and
//   g(t) = integral_t^inf (x - t) h(x) dx.
// Show that g''(t) = h(t).
//
// Solution. Write g(t) = integral_t^inf x h(x) dx - t integral_t^inf h(x) dx.
// Both integrals converge: |x h(x)| is integrable, and so is h on [t, inf)
// (|h| <= |x h| for |x| >= 1, and h is continuous on the rest). Then, by the
// fundamental theorem of calculus (the lower limit is t),
//   g'(t) = -t h(t) - integral_t^inf h(x) dx + t h(t) = -integral_t^inf h(x)
//   dx, g''(t) = h(t).
// (Or by the Leibniz rule: the boundary term -(t - t) h(t) vanishes and
// d/dt (x - t) = -1.)
//
// Application (the note in the book): with h(x) = e^(-rT) f(x), f the
// density of S(T), the call price is C(K) = integral_K^inf (x - K) h(x) dx,
// so d^2C/dK^2 = e^(-rT) f(K). Below this is checked with Black-Scholes,
// where S(T) is lognormal:
//   f(x) = exp(-(ln(x/S) - (r - sigma^2/2) T)^2 / (2 sigma^2 T)) / (x sigma
//   sqrt(2 pi T)).

// g(t) for the standard normal density h = phi, in closed form:
//   integral_t^inf x phi(x) dx = phi(t),  integral_t^inf phi = 1 - N(t),
//   g(t) = phi(t) - t (1 - N(t)),  g'(t) = -(1 - N(t)),  g''(t) = phi(t).
[[nodiscard]] inline double Ex06GNormal(double t) {
  return norm_pdf(t) - t * norm_cdf(-t);
}
[[nodiscard]] inline double Ex06GNormalPrime(double t) { return -norm_cdf(-t); }

// g(t) for any h, by numerical integration on [t, inf).
template <std::invocable<double> H>
[[nodiscard]] double Ex06G(H&& h, double t, double tol = 1e-10) {
  return ImproperIntegral([&](double x) { return (x - t) * h(x); }, t,
                          kInfinity, tol)
      .value;
}

// Black-Scholes call price as a function of the strike, and the
// discounted lognormal density e^(-rT) f(K).
struct Ex06Market {
  double spot;
  double maturity;
  double sigma;
  double r;
};

[[nodiscard]] inline double Ex06Call(const Ex06Market& m, double strike) {
  const double sv = m.sigma * std::sqrt(m.maturity);
  const double d1 = (std::log(m.spot / strike) +
                     (m.r + 0.5 * m.sigma * m.sigma) * m.maturity) /
                    sv;
  return m.spot * norm_cdf(d1) -
         strike * std::exp(-m.r * m.maturity) * norm_cdf(d1 - sv);
}

[[nodiscard]] inline double Ex06DiscountedDensity(const Ex06Market& m,
                                                  double strike) {
  const double sv = m.sigma * std::sqrt(m.maturity);
  const double z = (std::log(strike / m.spot) -
                    (m.r - 0.5 * m.sigma * m.sigma) * m.maturity) /
                   sv;
  return std::exp(-m.r * m.maturity) * norm_pdf(z) / (strike * sv);
}

}  // namespace p4mfe::ch02::ex
