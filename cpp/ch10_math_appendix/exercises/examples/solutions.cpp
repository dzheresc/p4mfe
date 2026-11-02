// Prints the numerical answers and checks for the exercises of section 10.10.
//
//   ./build/cpp/ch10_math_appendix/exercises/ch10_exercises_solutions

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <p4mfe/ch10/exercises/ex01_parity.hpp>
#include <p4mfe/ch10/exercises/ex02_lhopital_sums.hpp>
#include <p4mfe/ch10/exercises/ex03_sum_k4.hpp>
#include <p4mfe/ch10/exercises/ex04_sum_squares_recursion.hpp>
#include <p4mfe/ch10/exercises/ex05_07_linear_recursions.hpp>
#include <p4mfe/ch10/exercises/ex08_double_root.hpp>
#include <p4mfe/ch10/exercises/ex09_10_big_o.hpp>
#include <p4mfe/ch10/exercises/ex11_binomial_identities.hpp>
#include <p4mfe/ch10/exercises/ex12_13_harmonic_series.hpp>
#include <p4mfe/ch10/exercises/ex14_power_series_radius.hpp>
#include <p4mfe/ch10/exercises/ex15_17_nested_limits.hpp>

using namespace p4mfe::ch10;
using namespace p4mfe::ch10::ex;

namespace {
void header(const char* title) { std::printf("\n== %s ==\n", title); }
const char* name(Parity p) {
  return p == Parity::Even ? "even" : p == Parity::Odd ? "odd" : "neither";
}
}  // namespace

int main() {
  header("Ex 1: parity of x^i f(x^j), f = sin");
  const std::array<double, 5> xs{0.2, 0.5, 1.0, 1.5, 2.0};
  for (unsigned i = 1; i <= 3; ++i) {
    std::printf("  ");
    for (unsigned j = 1; j <= 3; ++j)
      std::printf("i=%u j=%u: %-5s  ", i, j,
                  name(parity(
                      [&](double x) {
                        return ex01_h([](double y) { return std::sin(y); }, i,
                                      j, x);
                      },
                      xs, 1e-10)));
    std::printf("\n");
  }

  header("Ex 2: S(n, 2) and S(n, 3) by l'Hopital's rule at x = 1");
  for (unsigned n : {1u, 5u, 10u, 100u})
    std::printf(
        "  n = %-4u N2'''(1) = %-10lld S(n,2) = %-8lld S(n,3) = %lld\n", n,
        static_cast<long long>(ex02_derivative_at_one(ex02_numerator(n, 2), 3)),
        static_cast<long long>(ex02_sum_by_lhopital(n, 2)),
        static_cast<long long>(ex02_sum_by_lhopital(n, 3)));

  header("Ex 3: S(n, 4) = n(n+1)(2n+1)(3n^2+3n-1)/30");
  for (unsigned n : {1u, 2u, 3u, 10u, 100u})
    std::printf("  n = %-4u closed form %-12llu recursion %llu\n", n,
                static_cast<unsigned long long>(ex03_sum_k4(n)),
                static_cast<unsigned long long>(ex03_sum_k4_recursive(n)));

  header("Ex 4: x_{n+4} - 4x_{n+3} + 6x_{n+2} - 4x_{n+1} + x_n = 0");
  const auto x4 = ex04_terms(8);
  std::printf("  x_1..x_8:");
  for (double v : x4) std::printf(" %.0f", v);
  const Vector c4 = ex04_cubic();
  std::printf(
      "\n  cubic coefficients c_0..c_3: %.6f %.6f %.6f %.6f  (0, 1/6, 1/2, "
      "1/3)\n",
      c4[0], c4[1], c4[2], c4[3]);

  header("Ex 5: x_{n+3} = 2x_{n+1} + x_n, x_0 = 1, x_1 = -1, x_2 = 1");
  const Vector c5 = ex05_constants();
  std::printf(
      "  c1 (-1)^n + c2 phi^n + c3 psi^n with (c1, c2, c3) = (%.3g, %.3g, "
      "%.3g): x_n = (-1)^n\n",
      c5[0], c5[1] + 0.0, c5[2] + 0.0);

  header("Ex 6: x_{n+1} = 3x_n + 2, x_0 = 1");
  std::printf("  x_n = 2 * 3^n - 1:");
  for (unsigned n = 0; n <= 6; ++n) std::printf(" %.0f", ex06_closed_form(n));
  std::printf("\n");

  header("Ex 7: x_{n+1} = 3x_n + n + 2, x_0 = 1");
  const Vector c7 = ex07_constants();
  std::printf(
      "  x_n = c1 + c2 n + c3 3^n, (c1, c2, c3) = (%.4f, %.4f, %.4f) = (-5/4, "
      "-1/2, 9/4)\n",
      c7[0], c7[1], c7[2]);
  std::printf("  x_n = (3^(n+2) - 2n - 5)/4:");
  for (unsigned n = 0; n <= 6; ++n) std::printf(" %.0f", ex07_closed_form(n));
  std::printf("\n");

  header("Ex 8: P(z) = (z-2)^2 (z+1), y_n = n 2^n");
  const std::array<double, 4> a8{4.0, 0.0, -3.0, 1.0};
  for (unsigned n : {0u, 5u, 10u})
    std::printf(
        "  n = %-3u residual at lambda = 2: %.3g, at the simple root -1: %.3g "
        "(= C lambda^(n+1) P'(lambda))\n",
        n, ex08_residual(a8, 2.0, 1.0, n), ex08_residual(a8, -1.0, 1.0, n));

  header(
      "Ex 9: |(f+g)/x^2| for f = x^2 sin(1/x), g = 3x^2, and for f = x^3, g = "
      "x^2.5");
  for (double x : {1e-1, 1e-2, 1e-3, 1e-4})
    std::printf("  x = %-6g O-sum ratio %.4f   o-sum ratio %.4g\n", x,
                std::abs(ex09_f_big(x) + ex09_g_big(x)) / (x * x),
                (ex09_f_small(x) + ex09_g_small(x)) / (x * x));

  header("Ex 10: ratios that stay bounded");
  std::printf("  %-8s %-12s %-14s %-12s %-14s\n", "n", "S2/n^3",
              "(S2-n^3/3)/n^2", "S3/n^4", "(S3-n^4/4)/n^3");
  for (unsigned n : {1u, 10u, 100u, 1000u, 10000u})
    std::printf("  %-8u %-12.6f %-14.6f %-12.6f %-14.6f\n", n, ex10_ratio_k2(n),
                ex10_remainder_ratio_k2(n), ex10_ratio_k3(n),
                ex10_remainder_ratio_k3(n));

  header("Ex 11: binomial identities");
  for (unsigned n : {1u, 5u, 10u, 20u})
    std::printf(
        "  n = %-3u sum C = %-8llu (2^n)  sum C(n,2k) = %-7llu (2^(n-1))  sum "
        "j C = %llu (n 2^(n-1))\n",
        n, static_cast<unsigned long long>(ex11_sum_binomials(n)),
        static_cast<unsigned long long>(ex11_sum_even_binomials(n)),
        static_cast<unsigned long long>(ex11_sum_j_binomials(n)));

  header("Ex 12: partial sums");
  for (unsigned n : {10u, 1000u, 1000000u}) {
    const auto [hl, hh] = ex12_harmonic_bounds(n);
    const auto [kl, kh] = ex12_k_log_k_bounds(n);
    std::printf(
        "  n = %-8u sum 1/k^2 = %.8f (pi^2/6 = %.8f)  %.4f < H_n = %.4f < %.4f "
        " %.4f < sum 1/(k ln k) = %.4f < %.4f\n",
        n, ex12_sum_inv_squares(n), std::numbers::pi * std::numbers::pi / 6.0,
        hl, ex12_harmonic(n), hh, kl, ex12_sum_inv_k_log_k(n), kh);
  }

  header("Ex 13: x_n = H_n - ln n");
  for (unsigned n : {1u, 2u, 10u, 1000u, 1000000u})
    std::printf("  n = %-8u x_n = %.10f  x_n - gamma = %.3e  1/(2n) = %.3e\n",
                n, ex13_x(n), ex13_x(n) - ex13_euler_gamma, 0.5 / n);

  header("Ex 14: sum x^k / (k ln k), R = 1");
  for (unsigned k : {10u, 1000u, 1000000u})
    std::printf("  |a_k/a_(k+1)| at k = %-8u %.8f\n", k, ex14_ratio(k));
  std::printf(
      "  x = 1:  S_n = %.4f, %.4f, %.4f for n = 10^2, 10^4, 10^6 (diverges "
      "like ln ln n)\n",
      ex14_partial_sum(1.0, 100), ex14_partial_sum(1.0, 10000),
      ex14_partial_sum(1.0, 1000000));
  std::printf(
      "  x = -1: S_n = %.8f, %.8f for n = 10^6, 10^6 + 1 (converges, "
      "conditionally)\n",
      ex14_partial_sum(-1.0, 1000000), ex14_partial_sum(-1.0, 1000001));

  header("Ex 15: sqrt(a + sqrt(a + ...)) = (1 + sqrt(1 + 4a))/2");
  for (double a : {1.0, 2.0, 6.0, 10.0})
    std::printf("  a = %-4g limit %.12f  50 iterations %.12f\n", a,
                ex15_limit(a), ex15_iterate(a, 50));

  header("Ex 16: a + 1/(a + 1/(a + ...)) = (a + sqrt(a^2 + 4))/2");
  for (double a : {0.5, 1.0, 2.0, 10.0})
    std::printf("  a = %-4g limit %.12f  50 iterations %.12f\n", a,
                ex16_limit(a), ex16_iterate(a, 50));

  header("Ex 17: power towers");
  std::printf("  (i)  x = sqrt 2 = %.12f, tower after 100 levels %.12f\n",
              ex17_x_for(2.0), ex17_tower(ex17_x_for(2.0), 100));
  std::printf("  (ii) largest x = e^(1/e) = %.12f, largest b = e = %.12f\n",
              ex17_x_max, std::numbers::e);
  for (unsigned n : {100u, 10000u, 1000000u})
    std::printf(
        "       tower of e^(1/e) after %-7u levels %.10f  (e - t_n) n / (2e) = "
        "%.4f\n",
        n, ex17_tower(ex17_x_max, n),
        (std::numbers::e - ex17_tower(ex17_x_max, n)) * n /
            (2.0 * std::numbers::e));
  return 0;
}
