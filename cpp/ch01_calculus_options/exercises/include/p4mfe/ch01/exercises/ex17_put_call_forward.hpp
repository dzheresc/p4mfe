#pragma once
// Exercise 17 (section 1.12): when are a call and a put worth the same?
//
// Show that the values of a plain vanilla put and a plain vanilla call with
// the same maturity and strike, on the same underlying asset, are equal if
// and only if the strike is equal to the forward price.
//
// Solution. Put-Call parity (1.9) with continuous dividend yield q:
//   C - P = S e^(-qT) - K e^(-rT) = e^(-rT) (S e^((r-q)T) - K) = e^(-rT) (F -
//   K),
// where F = S e^((r-q)T) is the forward price (1.10). Since e^(-rT) > 0,
// C = P  <=>  K = F.  (Same with discrete dividends: C - P = e^(-rT)(F - K)
// with F = (S - PV(dividends)) e^(rT).) Moreover C > P exactly when K < F.

#include <cmath>
#include <p4mfe/ch01/forwards.hpp>
#include <p4mfe/ch01/options.hpp>

namespace p4mfe::ch01::ex {

// C - P implied by parity, written through the forward price.
[[nodiscard]] inline double Ex17CallMinusPut(double spot, double strike,
                                             double maturity, double r,
                                             double q = 0.0) {
  return std::exp(-r * maturity) *
         (ForwardPrice(spot, maturity, r, q) - strike);
}

// The unique strike at which C = P.
[[nodiscard]] inline double Ex17EqualValueStrike(double spot, double maturity,
                                                 double r, double q = 0.0) {
  return ForwardPrice(spot, maturity, r, q);
}

}  // namespace p4mfe::ch01::ex
