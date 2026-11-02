#pragma once
// Chapter 8, section 8.5: Theta of a derivative security.
//
// A European derivative's value depends on calendar time t only through the
// time to maturity tau = T - t, so by the chain rule
//   Theta = dV/dt = dV/dtau * dtau/dt = -dV/dtau.
// Under risk-neutral valuation (chapter 4), with payoff h,
//   V(tau) = e^(-r tau) integral h(S e^((r - q - sigma^2/2) tau + sigma
//   sqrt(tau) z)) phi(z) dz,
// and dV/dtau is obtained by differentiating under the integral sign
// (chapter 2's improper_integral_derivative). For calls and puts this gives
// back chapter 3's closed-form Theta.

#include <cmath>
#include <concepts>
#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch02/integration.hpp>
#include <p4mfe/ch04/asset_model.hpp>
#include <p4mfe/normal.hpp>

namespace p4mfe::ch08 {

// ---------------------------------------------------------------- 8.5 ------
// Theta = -dV/dtau for a value given as a function of tau.
template <std::invocable<double> V>
[[nodiscard]] double ThetaFromMaturity(V&& v_of_tau, double tau) {
  return -ch01::Derivative(v_of_tau, tau);
}

// Theta of the European derivative with payoff h, by differentiating the
// risk-neutral integral under the integral sign in tau.
template <std::invocable<double> Payoff>
[[nodiscard]] double ThetaByIntegral(Payoff&& h, double spot, double maturity,
                                     double sigma, double r, double q = 0.0,
                                     double tol = 1e-7) {
  const auto integrand = [&](double z, double tau) {
    const double p = norm_pdf(z);
    return p == 0.0
               ? 0.0
               : std::exp(-r * tau) *
                     h(ch04::PriceFromNormal(z, spot, tau, sigma, r, q)) * p;
  };
  return -ch02::ImproperIntegralDerivative(integrand, -ch02::kInfinity,
                                           ch02::kInfinity, maturity, tol);
}

}  // namespace p4mfe::ch08
