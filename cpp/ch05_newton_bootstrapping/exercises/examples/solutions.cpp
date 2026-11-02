// Prints the numerical answers to the exercises of section 5.7.
//
//   ./build/cpp/ch05_newton_bootstrapping/exercises/ch05_exercises_solutions

#include <cstdio>
#include <p4mfe/ch05/exercises/ex01_bond_yield.hpp>
#include <p4mfe/ch05/exercises/ex02_zero_curve_bond.hpp>
#include <p4mfe/ch05/exercises/ex03_implied_vol_limits.hpp>
#include <p4mfe/ch05/exercises/ex04_implied_vol_methods.hpp>
#include <p4mfe/ch05/exercises/ex05_central_difference_newton.hpp>
#include <p4mfe/ch05/exercises/ex06_atm_delta_dividend.hpp>
#include <p4mfe/ch05/exercises/ex07_treasury_bootstrap.hpp>
#include <p4mfe/ch05/exercises/ex08_bootstrap_overnight.hpp>
#include <p4mfe/ch05/exercises/ex09_bootstrap_pricing_error.hpp>

using namespace p4mfe;
using namespace p4mfe::ch05;
using namespace p4mfe::ch05::ex;

namespace {
void header(const char* title) { std::printf("\n== %s ==\n", title); }
void print_curve(const char* label, const ch02::ZeroCurve& c) {
  std::printf("  %-6s", label);
  for (std::size_t i = 0; i < c.times.size(); ++i)
    std::printf("  r(%.4g) = %.6f%%", c.times[i], 100 * c.rates[i]);
  std::printf("\n");
}
}  // namespace

int main() {
  header("Ex 1: yield, duration, convexity");
  const auto e1 = ex01_solve();
  std::printf("  y = %.6f (%zu Newton steps)  D = %.6f  C = %.6f\n", e1.yield,
              e1.iterations, e1.duration, e1.convexity);

  header("Ex 2: bond off r(0,t) = 0.05 + 0.01 ln(1 + t/2)");
  const auto e2 = ex02_solve();
  std::printf("  B = %.6f  y = %.6f  D = %.6f  C = %.6f\n", e2.price, e2.yield,
              e2.duration, e2.convexity);
  std::printf("  semiannual yield %.6f, modified duration %.6f\n",
              e2.yield_semiannual, e2.modified_duration_semiannual);

  header("Ex 3: limits of f (market of Ex 4)");
  const Ex04Market m4;
  const Ex03Market m3{m4.S, m4.K, m4.T, m4.r, m4.q, m4.C};
  std::printf(
      "  F = %.6f  lim_0 f = %.6f  lim_inf f = %.6f  f(1e-4) = %.6f  f(50) = "
      "%.6f\n",
      ex03_forward(m3), ex03_limit_zero(m3), ex03_limit_infinity(m3),
      ex03_f(1e-4, m3), ex03_f(50.0, m3));
  const auto rg = ex03_price_range(m3.S, m3.K, m3.T, m3.r, m3.q);
  std::printf("  implied vol exists for %.6f < C < %.6f\n", rg.lower, rg.upper);

  header("Ex 4: implied volatility");
  const auto b = ex04_bisection(), s = ex04_secant(), n = ex04_newton();
  std::printf("  bisection %.9f (%zu)  secant %.9f (%zu)  Newton %.9f (%zu)\n",
              b.x, b.iterations, s.x, s.iterations, n.x, n.iterations);
  std::printf("  Newton iterates:");
  for (double x : n.iterates) std::printf(" %.9f", x);
  std::printf("\n  secant iterates:");
  for (double x : s.iterates) std::printf(" %.9f", x);
  std::printf("\n  sigma_approx = %.6f  relative error %.3e\n",
              ex04_sigma_approx(), ex04_relative_error());

  header("Ex 5: approximate Newton, central vs forward differences");
  for (const Vector& x0 : {Vector{1, 2, 3}, Vector{2, 2, 2}}) {
    const auto c = ex05_solve(x0, Difference::Central);
    const auto f = ex05_solve(x0, Difference::Forward);
    const auto e = ex05_solve_exact(x0);
    std::printf("  x0 = (%g, %g, %g)\n", x0[0], x0[1], x0[2]);
    std::printf("    central  (%.9f, %.9f, %.9f) in %zu\n", c.x[0], c.x[1],
                c.x[2], c.iterations);
    std::printf("    forward  (%.9f, %.9f, %.9f) in %zu\n", f.x[0], f.x[1],
                f.x[2], f.iterations);
    std::printf("    exact J  (%.9f, %.9f, %.9f) in %zu\n", e.x[0], e.x[1],
                e.x[2], e.iterations);
  }

  header("Ex 6: q with Delta(ATM call) = 0.5");
  const auto q = ex06_solve();
  std::printf(
      "  q = %.9f after %zu Newton steps; r + sigma^2/2 = %.6f\n  iterates:",
      q.x, q.iterations, ex06_upper_bound());
  for (double x : q.iterates) std::printf(" %.9f", x);
  std::printf("\n");

  header("Ex 7: Treasury bootstrap");
  const auto c7 = ex07_curve_continuous();
  const auto s7 = ex07_curve_semiannual();
  print_curve("cont", c7);
  print_curve("semi", s7);
  std::printf("  semi - converted cont (bp):");
  for (std::size_t i = 0; i < c7.times.size(); ++i)
    std::printf(" %.2e",
                1e4 * (s7.rates[i] - ex07_to_compounded(c7.rates[i], 2)));
  std::printf("\n  semi - cont (bp):");
  for (std::size_t i = 0; i < c7.times.size(); ++i)
    std::printf(" %.2f", 1e4 * (s7.rates[i] - c7.rates[i]));
  std::printf("\n");

  header("Ex 8: bootstrap with overnight rate");
  const auto c8 = ex08_curve();
  print_curve("", c8);
  std::printf("  residual of (5.43): %.2e\n", ex08_hint_residual(c8));

  header("Ex 9: bootstrap and price the 3-year bond");
  const auto c9 = ex09_curve();
  print_curve("", c9);
  std::printf("  r(0, 3) interpolated = %.6f%%\n",
              100 * zero_rate_linear(c9, 3.0));
  const auto cf3 = ch02::bond_cash_flows(3.0, 0.045, 2);
  std::printf(
      "  model price %.6f  market %.6f  relative error %.6f%%  (3y yield "
      "%.6f%%)\n",
      ex09_model_price(c9), ex09_three_year.price,
      100 * ex09_relative_error(c9),
      100 * bond_yield_newton(cf3, ex09_three_year.price).x);
}
