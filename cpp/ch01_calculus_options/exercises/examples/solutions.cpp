// Prints the numerical answers to the exercises of section 1.12 and ASCII
// payoff diagrams for the option strategies (exercises 18-20, 26, 28).
//
//   ./build/cpp/ch01_calculus_options/exercises/ch01_exercises_solutions

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <p4mfe/ch01/exercises/ex01_06_antiderivatives.hpp>
#include <p4mfe/ch01/exercises/ex07_10_limits.hpp>
#include <p4mfe/ch01/exercises/ex11_delta_integral.hpp>
#include <p4mfe/ch01/exercises/ex12_13_integral_limits.hpp>
#include <p4mfe/ch01/exercises/ex14_bond_yield_derivatives.hpp>
#include <p4mfe/ch01/exercises/ex15_quadratic_taylor.hpp>
#include <p4mfe/ch01/exercises/ex16_heat_equation.hpp>
#include <p4mfe/ch01/exercises/ex17_put_call_forward.hpp>
#include <p4mfe/ch01/exercises/ex18_21_spreads.hpp>
#include <p4mfe/ch01/exercises/ex22_24_parity_arbitrage.hpp>
#include <p4mfe/ch01/exercises/ex25_27_trading_strategies.hpp>
#include <p4mfe/ch01/exercises/ex28_29_call_convexity.hpp>
#include <p4mfe/ch01/exercises/ex30_futures_arbitrage.hpp>
#include <string>
#include <vector>

using namespace p4mfe::ch01;
using namespace p4mfe::ch01::ex;

namespace {

void header(const char* title) { std::printf("\n== %s ==\n", title); }

// One row per S(T): value and a bar of '#' (or '-' for negative values).
void diagram(const char* name, const Portfolio& p, double lo, double hi,
             std::size_t n, bool pnl = false) {
  std::printf("%s\n   S(T)    %s\n", name, pnl ? "P&L" : "V(T)");
  const auto rows = payoff_table(p, lo, hi, n);
  double vmax = 1e-12;
  for (const auto& [s, v0] : rows)
    vmax = std::max(vmax, std::abs(pnl ? v0 - p.cost() : v0));
  const double scale = 30.0 / vmax;  // longest bar: 30 characters
  for (const auto& [s, v0] : rows) {
    const double v = pnl ? v0 - p.cost() : v0;
    const auto len = static_cast<std::size_t>(std::lround(std::abs(v) * scale));
    std::printf("  %6.2f %7.2f  %s\n", s, v,
                std::string(len, v < 0 ? '-' : '#').c_str());
  }
}

}  // namespace

int main() {
  header("Ex 1-6: antiderivatives (F(2) - F(1))");
  std::printf("  1  int_1^2 ln x dx            = %.10f\n",
              ex01_ln_antiderivative(2) - ex01_ln_antiderivative(1));
  std::printf(
      "  2  int_2^3 dx/(x ln x)        = %.10f\n",
      ex02_inv_x_ln_x_antiderivative(3) - ex02_inv_x_ln_x_antiderivative(2));
  std::printf("  3  int_0^1 dx/(1+x^2)         = %.10f  (pi/4)\n",
              ex03_inv_1_plus_x2_antiderivative(1));
  std::printf("  4  int_1^2 x^3 ln x dx        = %.10f\n",
              ex04_x_pow_n_ln_antiderivative(3, 2) -
                  ex04_x_pow_n_ln_antiderivative(3, 1));
  std::printf("  5  int_1^2 x^3 e^x dx         = %.10f\n",
              ex05_x_pow_n_exp_antiderivative(3, 2) -
                  ex05_x_pow_n_exp_antiderivative(3, 1));
  std::printf(
      "  6  int_1^2 (ln x)^3 dx        = %.10f\n",
      ex06_ln_pow_n_antiderivative(3, 2) - ex06_ln_pow_n_antiderivative(3, 1));

  header("Ex 7-10: limits");
  for (double x : {1.0, 10.0, 1000.0}) {
    const auto [lo, hi] = ex07_e_bounds(x);
    std::printf("  7  x = %6.0f: %.8f < e < %.8f\n", x, lo, hi);
  }
  std::printf(
      "  8  sqrt quotient at 1e-6 = %.10f (limit -1/8); exp quotient = %.10f "
      "(limit 1/6)\n",
      ex08_sqrt_quotient(1e-6), ex08_exp_quotient(1e-6));
  std::printf(
      "  9  (i) at 1e8 = %.10f (limit -1/2); (ii) at 1e8 = %.4e (-> -inf)\n",
      ex09_expr_i(1e8), ex09_expr_ii(1e8));
  std::printf(" 10  (1 - 1/x)^x at 1e8 = %.10f, 1/e = %.10f\n", ex10_expr(1e8),
              std::exp(-1.0));

  header("Ex 11: g'(x) for K = 100, T = 0.5, sigma = 0.3, r = 0.04");
  const Ex11Params p11{100, 0.5, 0.3, 0.04};
  for (double x : {80.0, 100.0, 120.0})
    std::printf("  x = %5.1f: g'(x) = %.8f\n", x, ex11_g_prime(p11, x));

  header("Ex 12-13: limits of integrals");
  for (double h : {0.1, 0.01}) {
    std::printf(
        " 12  h = %.2f: average of cos on [1-h, 1+h] - cos(1) = %.3e\n", h,
        ex12_local_average([](double x) { return std::cos(x); }, 1.0, h) -
            std::cos(1.0));
  }
  for (double s : {0.1, 0.01}) {
    std::printf(
        " 13  sigma = %.2f: E[cos(0.5 + sigma Z)] - cos(0.5) = %.3e\n", s,
        ex13_gaussian_average([](double x) { return std::cos(x); }, 0.5, s) -
            std::cos(0.5));
  }

  header("Ex 14: 2-year 5% semiannual bond at y = 6%");
  std::vector<CashFlow> cfs{{0.5, 2.5}, {1.0, 2.5}, {1.5, 2.5}, {2.0, 102.5}};
  const auto f = ex14_continuous(cfs, 0.06);
  const auto g = ex14_discrete(cfs, 0.06, 2);
  std::printf("  continuous: f = %.6f, f' = %.6f, f'' = %.6f\n", f.value,
              f.first, f.second);
  std::printf("  semiannual: g = %.6f, g' = %.6f, g'' = %.6f\n", g.value,
              g.first, g.second);

  header("Ex 15: f at a = (1, -1, 0)");
  const auto grad = ex15_gradient(ex15_a);
  std::printf(
      "  f(a) = %.0f, Df(a) = (%.0f, %.0f, %.0f), D2f = "
      "[[4,-1,0],[-1,0,3],[0,3,-2]]\n",
      ex15_f(ex15_a), grad[0], grad[1], grad[2]);

  header("Ex 16: u_t - u_xx at (x, t) = (0.7, 0.3)");
  std::printf("  %.3e\n", ex16_u_t(0.7, 0.3) - ex16_u_xx(0.7, 0.3));

  header("Ex 17: C = P iff K = F");
  std::printf("  S = 50, T = 1, r = 5%%, q = 2%%: F = %.4f\n",
              ex17_equal_value_strike(50, 1, 0.05, 0.02));

  header("Ex 18-21: spreads");
  diagram("18  butterfly 30/35/40", ex18_butterfly(), 26, 44, 18);
  diagram("19  bull call spread 30/35", ex19_bull_spread(), 26, 40, 14);
  diagram("19  bear put spread 20/15", ex19_bear_spread(), 10, 24, 14);
  std::printf(
      "20  butterfly 45/50/55 costs %.2f, profitable for %.0f < S(T) < %.0f\n",
      ex20_butterfly().cost(), ex20_breakeven_low, ex20_breakeven_high);
  diagram("20  P&L", ex20_butterfly(), 42, 58, 16, true);
  std::printf(
      "21  Portfolio 1 - Portfolio 2 payoff at S = X: %.1f (never negative)\n",
      ex21_portfolio1(50).payoff(50) - ex21_portfolio2(50).payoff(50));

  header("Ex 22-24: Put-Call parity arbitrage");
  if (const auto a = ex22_arbitrage()) {
    std::printf(
        " 22  calls %+.0f, puts %+.0f, shares %+.5f, cash %+.4f -> profit at T "
        "%.4f\n",
        a->calls, a->puts, a->shares, a->cash, a->profit);
  }
  std::printf(" 23  arbitrage: %s\n", ex23_arbitrage() ? "yes" : "no");

  header("Ex 25-27: strategies");
  std::printf(" 25  crossover S = %.3f\n", ex25_crossover);
  for (double s : {30.0, 35.0, 39.0, 40.0, 42.5, 45.0})
    std::printf("     S(T) = %4.1f: stock %+7.1f%%, calls %+7.1f%%\n", s,
                100 * ex25_stock_return(s), 100 * ex25_call_return(s));
  const auto p26 = ex26_portfolio();
  std::printf(" 26  cash %.0f, shares %.0f", p26.cash, p26.shares);
  for (const auto& leg : p26.options)
    std::printf(", %+.0f calls K = %.0f", leg.quantity, leg.strike);
  std::printf("\n");
  diagram("     V(T)", p26, 0, 60, 12);
  const auto p27 = ex27_range_digital(10, 40, 60, 0.5);
  std::printf(" 27  c = 10 on [40, 60], eps = 0.5:");
  for (const auto& leg : p27.options)
    std::printf(" %+.0f C(%.1f)", leg.quantity, leg.strike);
  std::printf("\n");

  header("Ex 28-30: arbitrage");
  const auto x28 = ex28_data();
  std::printf(" 28  convexity gap %.4f (< 0: arbitrage)\n",
              ex29_convexity_gap(x28));
  if (const auto a = ex29_arbitrage(x28)) {
    for (const auto& leg : a->options)
      std::printf("     %+.0f calls K = %.0f\n", leg.quantity, leg.strike);
    std::printf("     cash collected now %.2f\n", -a->cost());
    diagram("     payoff", *a, 95, 135, 8);
  }
  if (const auto a = ex30_arbitrage()) {
    std::printf(
        " 30  futures %+.0f, index units %+.5f, cash lent %.2f -> profit at T "
        "%.2f\n",
        a->futures, a->units, a->cash, a->profit);
  }
}
