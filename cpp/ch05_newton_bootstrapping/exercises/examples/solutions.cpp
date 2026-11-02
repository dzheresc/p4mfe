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
void Header(const char* title) { std::printf("\n== %s ==\n", title); }
void PrintCurve(const char* label, const ch02::ZeroCurve& c) {
  std::printf("  %-6s", label);
  for (std::size_t i = 0; i < c.times.size(); ++i)
    std::printf("  r(%.4g) = %.6f%%", c.times[i], 100 * c.rates[i]);
  std::printf("\n");
}
}  // namespace

int main() {
  Header("Ex 1: yield, duration, convexity");
  const auto e1 = Ex01Solve();
  std::printf("  y = %.6f (%zu Newton steps)  D = %.6f  C = %.6f\n", e1.yield,
              e1.iterations, e1.duration, e1.convexity);

  Header("Ex 2: bond off r(0,t) = 0.05 + 0.01 ln(1 + t/2)");
  const auto e2 = Ex02Solve();
  std::printf("  B = %.6f  y = %.6f  D = %.6f  C = %.6f\n", e2.price, e2.yield,
              e2.duration, e2.convexity);
  std::printf("  semiannual yield %.6f, modified duration %.6f\n",
              e2.yield_semiannual, e2.modified_duration_semiannual);

  Header("Ex 3: limits of f (market of Ex 4)");
  const Ex04Market m4;
  const Ex03Market m3{m4.spot, m4.strike, m4.maturity, m4.r, m4.q, m4.call};
  std::printf(
      "  F = %.6f  lim_0 f = %.6f  lim_inf f = %.6f  f(1e-4) = %.6f  f(50) = "
      "%.6f\n",
      Ex03Forward(m3), Ex03LimitZero(m3), Ex03LimitInfinity(m3),
      Ex03F(1e-4, m3), Ex03F(50.0, m3));
  const auto rg = Ex03PriceRange(m3.spot, m3.strike, m3.maturity, m3.r, m3.q);
  std::printf("  implied vol exists for %.6f < C < %.6f\n", rg.lower, rg.upper);

  Header("Ex 4: implied volatility");
  const auto b = Ex04Bisection(), s = Ex04Secant(), n = Ex04Newton();
  std::printf("  bisection %.9f (%zu)  secant %.9f (%zu)  Newton %.9f (%zu)\n",
              b.x, b.iterations, s.x, s.iterations, n.x, n.iterations);
  std::printf("  Newton iterates:");
  for (double x : n.iterates) std::printf(" %.9f", x);
  std::printf("\n  secant iterates:");
  for (double x : s.iterates) std::printf(" %.9f", x);
  std::printf("\n  sigma_approx = %.6f  relative error %.3e\n",
              Ex04SigmaApprox(), Ex04RelativeError());

  Header("Ex 5: approximate Newton, central vs forward differences");
  for (const Vector& x0 : {Vector{1, 2, 3}, Vector{2, 2, 2}}) {
    const auto c = Ex05Solve(x0, Difference::kCentral);
    const auto f = Ex05Solve(x0, Difference::kForward);
    const auto e = Ex05SolveExact(x0);
    std::printf("  x0 = (%g, %g, %g)\n", x0[0], x0[1], x0[2]);
    std::printf("    central  (%.9f, %.9f, %.9f) in %zu\n", c.x[0], c.x[1],
                c.x[2], c.iterations);
    std::printf("    forward  (%.9f, %.9f, %.9f) in %zu\n", f.x[0], f.x[1],
                f.x[2], f.iterations);
    std::printf("    exact J  (%.9f, %.9f, %.9f) in %zu\n", e.x[0], e.x[1],
                e.x[2], e.iterations);
  }

  Header("Ex 6: q with Delta(ATM call) = 0.5");
  const auto q = Ex06Solve();
  std::printf(
      "  q = %.9f after %zu Newton steps; r + sigma^2/2 = %.6f\n  iterates:",
      q.x, q.iterations, Ex06UpperBound());
  for (double x : q.iterates) std::printf(" %.9f", x);
  std::printf("\n");

  Header("Ex 7: Treasury bootstrap");
  const auto c7 = Ex07CurveContinuous();
  const auto s7 = Ex07CurveSemiannual();
  PrintCurve("cont", c7);
  PrintCurve("semi", s7);
  std::printf("  semi - converted cont (bp):");
  for (std::size_t i = 0; i < c7.times.size(); ++i)
    std::printf(" %.2e",
                1e4 * (s7.rates[i] - Ex07ToCompounded(c7.rates[i], 2)));
  std::printf("\n  semi - cont (bp):");
  for (std::size_t i = 0; i < c7.times.size(); ++i)
    std::printf(" %.2f", 1e4 * (s7.rates[i] - c7.rates[i]));
  std::printf("\n");

  Header("Ex 8: bootstrap with overnight rate");
  const auto c8 = Ex08Curve();
  PrintCurve("", c8);
  std::printf("  residual of (5.43): %.2e\n", Ex08HintResidual(c8));

  Header("Ex 9: bootstrap and price the 3-year bond");
  const auto c9 = Ex09Curve();
  PrintCurve("", c9);
  std::printf("  r(0, 3) interpolated = %.6f%%\n",
              100 * ZeroRateLinear(c9, 3.0));
  const auto cf3 = ch02::BondCashFlows(3.0, 0.045, 2);
  std::printf(
      "  model price %.6f  market %.6f  relative error %.6f%%  (3y yield "
      "%.6f%%)\n",
      Ex09ModelPrice(c9), kEx09ThreeYear.price, 100 * Ex09RelativeError(c9),
      100 * BondYieldNewton(cf3, kEx09ThreeYear.price).x);
}
