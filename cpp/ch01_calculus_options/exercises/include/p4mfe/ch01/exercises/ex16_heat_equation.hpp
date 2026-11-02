#pragma once
// Exercise 16 (section 1.12): the fundamental solution of the heat equation.
//
// Let u(x, t) = 1/sqrt(4 pi t) e^(-x^2/(4t)),  t > 0, x real.
// Compute du/dt and d^2u/dx^2, and show that du/dt = d^2u/dx^2.
// (Note: u is the density of a normal variable with mean 0 and variance 2t,
//  and is used in the PDE derivation of the Black-Scholes formula.)
//
// Solution. Write u = (4 pi t)^(-1/2) e^(-x^2/(4t)).
//   du/dt   = (4 pi t)^(-1/2) e^(...) * ( -1/(2t) + x^2/(4t^2) )
//           = u * (x^2 - 2t) / (4 t^2);
//   du/dx   = u * (-x/(2t));
//   d2u/dx2 = u * ( x^2/(4t^2) - 1/(2t) ) = u * (x^2 - 2t) / (4 t^2).
// The two agree, so u solves u_t = u_xx.

#include <cmath>
#include <numbers>

namespace p4mfe::ch01::ex {

[[nodiscard]] inline double ex16_u(double x, double t) {
  return std::exp(-x * x / (4.0 * t)) / std::sqrt(4.0 * std::numbers::pi * t);
}

[[nodiscard]] inline double ex16_u_t(double x, double t) {
  return ex16_u(x, t) * (x * x - 2.0 * t) / (4.0 * t * t);
}

[[nodiscard]] inline double ex16_u_x(double x, double t) {
  return ex16_u(x, t) * (-x / (2.0 * t));
}

[[nodiscard]] inline double ex16_u_xx(double x, double t) {
  return ex16_u(x, t) * (x * x - 2.0 * t) / (4.0 * t * t);
}

}  // namespace p4mfe::ch01::ex
