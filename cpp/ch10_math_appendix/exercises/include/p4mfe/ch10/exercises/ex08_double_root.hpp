#pragma once
// Exercise 8 (section 10.10): a double root lambda of the characteristic
// polynomial gives the solution C n lambda^n.

#include <p4mfe/ch10/interpolation.hpp>

#include <cmath>
#include <span>

namespace p4mfe::ch10::ex {

// ------------------------------------------------------------ Exercise 8 ---
// Let P(z) = sum_{i=0}^{k} a_i z^i be the characteristic polynomial of
//   sum_{i=0}^{k} a_i x_{n+i} = 0,  n >= 0  (10.96),
// and let lambda be a root of multiplicity 2 of P. Show that y_n = C n lambda^n
// satisfies (10.96).
//
// Proof. Split the factor n + i:
//   sum_i a_i y_{n+i} = C sum_i a_i (n + i) lambda^(n+i)
//                     = C n lambda^n sum_i a_i lambda^i + C lambda^(n+1) sum_i i a_i lambda^(i-1)
//                     = C n lambda^n P(lambda) + C lambda^(n+1) P'(lambda)
// for every n >= 0. lambda is a root of multiplicity (at least) 2 iff
// P(lambda) = 0 and P'(lambda) = 0 (P(z) = (z - lambda)^2 Q(z) gives
// P' = 2(z - lambda)Q + (z - lambda)^2 Q', which vanishes at lambda; conversely
// P(lambda) = 0 gives P = (z - lambda) R and P'(lambda) = R(lambda) = 0 gives
// R = (z - lambda) Q). So both terms vanish and y_n solves (10.96).
//
// The identity also shows what fails for a simple root: then P'(lambda) != 0
// and the residual is C lambda^(n+1) P'(lambda) != 0 (for lambda, C != 0).
// For a root of multiplicity m the same argument with
// (n + i)^r expanded gives the solutions n^r lambda^n, r < m.

// sum_{i=0}^{k} a_i y_{n+i} for y_m = C m lambda^m, computed directly.
[[nodiscard]] inline double ex08_residual(std::span<const double> a, double lambda, double C, unsigned n) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double m = static_cast<double>(n + i);
        s += a[i] * C * m * std::pow(lambda, m);
    }
    return s;
}

// The right side of the hint: C n lambda^n P(lambda) + C lambda^(n+1) P'(lambda).
[[nodiscard]] inline double ex08_hint(std::span<const double> a, double lambda, double C, unsigned n) {
    double dP = 0.0;
    for (std::size_t i = 1; i < a.size(); ++i) dP += static_cast<double>(i) * a[i] * std::pow(lambda, static_cast<double>(i) - 1.0);
    const double N = n;
    return C * N * std::pow(lambda, N) * polynomial_value(a, lambda) + C * std::pow(lambda, N + 1.0) * dP;
}

}  // namespace p4mfe::ch10::ex
