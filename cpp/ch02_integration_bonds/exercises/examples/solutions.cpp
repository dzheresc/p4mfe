// Prints the numerical answers to the exercises of section 2.12.
//
//   ./build/cpp/ch02_integration_bonds/exercises/ch02_exercises_solutions

#include <cmath>
#include <cstdio>
#include <p4mfe/ch02/exercises/ex01_02_improper_integrals.hpp>
#include <p4mfe/ch02/exercises/ex03_04_quadrature.hpp>
#include <p4mfe/ch02/exercises/ex05_06_integral_derivatives.hpp>
#include <p4mfe/ch02/exercises/ex07_simpson_identity.hpp>
#include <p4mfe/ch02/exercises/ex08_09_rate_curves.hpp>
#include <p4mfe/ch02/exercises/ex10_13_bond_pricing.hpp>
#include <p4mfe/ch02/exercises/ex14_19_duration_convexity.hpp>

using namespace p4mfe::ch02;
using namespace p4mfe::ch02::ex;

namespace {
void header(const char* title) { std::printf("\n== %s ==\n", title); }
void metrics(const char* name, const BondMetrics& m) {
  std::printf("  %s  B = %.4f  D = %.6f  C = %.6f\n", name, m.price, m.duration,
              m.convexity);
}
}  // namespace

int main() {
  header("Ex 1: lim (N(x) - 1/2 - x/sqrt(2pi)) / x^3");
  std::printf("  limit = -1/(6 sqrt(2pi)) = %.12f\n", ex01_limit);
  for (double x : {0.1, 1e-3, 1e-6})
    std::printf("  x = %-6g quotient = %.12f\n", x, ex01_quotient(x));

  header("Ex 2: Gamma function");
  for (double a : {0.5, 1.0, 2.5, 5.0})
    std::printf("  Gamma(%.1f) = %.10f  (std::tgamma %.10f)\n", a,
                ex02_gamma(a), std::tgamma(a));

  header("Ex 3: int_1^3 sqrt(x) e^-x dx, tol 1e-6");
  const char* names[] = {"Midpoint   ", "Trapezoidal", "Simpson    "};
  const Rule rules[] = {Rule::Midpoint, Rule::Trapezoidal, Rule::Simpson};
  for (int i = 0; i < 3; ++i) {
    const auto r = ex03_approx(rules[i]);
    std::printf("  %s %.9f  n = %zu\n", names[i], r.value, r.intervals);
  }
  std::printf("  exact       %.12f\n", ex03_exact());

  header("Ex 4: int_0^1 x^(5/2)/(1+x^2) dx");
  std::printf("  exact          %.12f\n", ex04_exact());
  const auto m4 = ex04_midpoint();
  std::printf("  (i) midpoint   %.9f  n = %zu\n", m4.value, m4.intervals);
  std::printf("  (ii) f''''(x) ~ -(15/16) x^(-3/2):  f''''(1e-6) = %.4g\n",
              ex04_f4(1e-6));
  std::printf(
      "  (iii) Simpson\n       n   S_n              error      order\n");
  double prev = 0.0;
  for (const auto& r : ex04_simpson_table()) {
    std::printf("    %4zu   %.12f  %.3e", r.n, r.value, r.error);
    if (prev > 0.0) std::printf("  %.3f", std::log2(prev / r.error));
    std::printf("\n");
    prev = r.error;
  }

  header("Ex 5: g'(x) = exp(-b(x)^2/2) / (x sigma sqrt(2 pi T))");
  const Ex05Params p5{100.0, 0.5, 0.3, 0.04};
  std::printf("  K = 100, T = 0.5, sigma = 0.3, r = 0.04: g'(100) = %.8f\n",
              ex05_g_prime(p5, 100.0));

  header("Ex 6: g''(t) = h(t); d2C/dK2 = e^(-rT) f(K)");
  const Ex06Market m6{100.0, 1.0, 0.25, 0.03};
  const double K = 100.0, h = 0.5;
  const double c2 =
      (ex06_call(m6, K + h) - 2 * ex06_call(m6, K) + ex06_call(m6, K - h)) /
      (h * h);
  std::printf("  BS, K = 100: d2C/dK2 ~ %.8f, e^(-rT) f(K) = %.8f\n", c2,
              ex06_discounted_density(m6, K));

  header("Ex 7: S_n = (2 M_n + T_n) / 3");
  const auto f7 = [](double x) { return std::exp(-x * x); };
  std::printf("  e^(-x^2) on [0, 2], n = 4: S = %.15f, (2M+T)/3 = %.15f\n",
              simpson_rule(f7, 0.0, 2.0, 4),
              ex07_simpson_from_mt(f7, 0.0, 2.0, 4));

  header("Ex 9: r_c(0,t) = 0.05 + 0.01 ln(1 + t/2)");
  std::printf("     t   r_c(0,t)   r(t)       annual     semiannual\n");
  for (double t : {0.0, 0.5, 1.0, 2.0, 5.0, 10.0})
    std::printf("  %4.1f   %.6f   %.6f   %.6f   %.6f\n", t, ex09_zero_rate(t),
                ex09_inst_rate(t), ex09_annual_zero_rate(t),
                ex09_semiannual_zero_rate(t));

  header("Ex 10-11: zero rates 5%, 5.25%, 5.35%, 5.5%");
  const auto d10 = ex10_discount_factors();
  std::printf("  discount factors %.6f %.6f %.6f %.6f\n", d10[0], d10[1],
              d10[2], d10[3]);
  std::printf("  10  price of 2y 5%% semiannual bond = %.4f\n", ex10_price());
  std::printf("  11  par yield = %.4f%%\n", 100 * ex11_par_yield());

  header("Ex 12: r(t) = 0.05 + 0.005 ln(1 + t)");
  for (double t : {0.5, 1.0, 1.5, 2.0})
    std::printf("  t = %.1f  r(0,t) = %.6f  D = %.6f\n", t, ex12_zero_rate(t),
                ex12_discount_factor(t));
  std::printf("  price of 2y 5%% semiannual bond = %.4f\n", ex12_price());

  header("Ex 13: r(t) = 0.05 / (1 + exp(-(1+t)^2))");
  for (double t : {1.0, 2.0, 3.0}) {
    const auto I = ex13_integral(t, ex13_tol(t));
    std::printf("  D(0,%.0f) = %.*f  (tol %.0e, n = %zu)\n", t, t > 2 ? 8 : 6,
                std::exp(-I.value), ex13_tol(t), I.intervals);
  }
  std::printf("  price of 3y 5%% annual bond = %.4f\n", ex13_price());

  header("Ex 14-15: price, duration, convexity");
  metrics("14  30m 6% semiannual, y = 9%  ", ex14_metrics());
  metrics("15  14m 8% quarterly,  y = 7%  ", ex15_metrics());

  header("Ex 16: 10y semiannual bond, y = 6%, rising coupon");
  for (double c : {0.0, 0.03, 0.06, 0.1}) {
    std::printf("  c = %4.1f%%", 100 * c);
    metrics("", ex16_metrics(10.0, c, 2, 0.06));
  }

  header("Ex 17: 10y zero, yield +10bp");
  std::printf("  duration: %.4f%%, with convexity: %.4f%%, exact: %.4f%%\n",
              100 * ex17_duration_estimate, 100 * ex17_convexity_estimate(),
              100 * ex17_exact_relative_change());

  header("Ex 18: B = 102, D = 3.5, yield -50bp");
  std::printf("  B_new ~ %.3f\n", ex18_new_price());

  header("Ex 19: C = D^2 - dD/dy (bond of Ex 14)");
  const auto cf = bond_cash_flows(2.5, 0.06, 2);
  const auto m = bond_metrics(cf, 0.09);
  std::printf("  C = %.8f,  D^2 - dD/dy = %.8f\n", m.convexity,
              m.duration * m.duration - ex19_duration_slope(cf, 0.09));
  return 0;
}
