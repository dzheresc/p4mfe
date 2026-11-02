// Sections 10.4-10.8: closed forms of linear recursions, Big O orders,
// Stirling's formula, and convergence of Taylor series.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch10/big_o.hpp>
#include <p4mfe/ch10/linear_recursions.hpp>
#include <p4mfe/ch10/power_series.hpp>
#include <p4mfe/ch10/stirling.hpp>
#include <p4mfe/ch10/taylor_convergence.hpp>
#include <vector>

int main() {
  using namespace p4mfe::ch10;

  std::printf("Fibonacci: x_n = c1 phi^n + c2 (1 - phi)^n\n");
  const auto fib = solve_second_order(1.0, 1.0, 0.0, 1.0);
  std::printf("  roots %.12f %.12f, c1 = %.12f, c2 = %.12f\n",
              fib.lambda1.real(), fib.lambda2.real(), fib.c1.real(),
              fib.c2.real());
  for (unsigned n : {10u, 20u, 50u, 70u})
    std::printf("  x_%u = %.1f\n", n, fib(n));

  std::printf("\nMortgage 300,000 at 6%%/12 for 360 months: payment %.2f\n",
              loan_payment(300000, 0.005, 360));

  std::printf("\nObserved orders: e^h - 1 - h = O(h^2), sin h - h = O(h^3)\n");
  for (double h : {1e-1, 1e-2, 1e-3})
    std::printf("  h = %.0e   %.4f   %.4f\n", h,
                observed_power([](double t) { return std::exp(t) - 1 - t; }, h),
                observed_power([](double t) { return std::sin(t) - t; }, h));

  std::printf(
      "\nStirling's formula: n! / (sqrt(2 pi n)(n/e)^n) - 1 vs. 1/(12n)\n");
  for (double n : {1.0, 2.0, 5.0, 10.0, 20.0, 100.0})
    std::printf("  n = %5.0f   %.6e   %.6e\n", n,
                std::tgamma(n + 1) / stirling(n) - 1, 1 / (12 * n));
  std::printf("  C(2n, n) sqrt(pi n) / 4^n at n = 400: %.8f\n",
              central_binomial(400) / central_binomial_stirling(400));

  std::printf("\nTaylor terms for e^x to 1e-12 (C = e^|x|, M = 1)\n");
  for (double x : {0.5, 1.0, 5.0, 10.0}) {
    const unsigned n = taylor_terms_needed(std::exp(std::abs(x)), 1.0, 0.0, x,
                                           1e-12 * std::exp(x));
    std::printf("  x = %4.1f  n = %2u  rel. error %.2e\n", x, n,
                exp_power_series(n)(x) / std::exp(x) - 1);
  }
  std::printf(
      "\ne^(-1/x^2): all Taylor coefficients at 0 vanish, but f(0.5) = %.6f\n",
      flat_function(0.5));
}
