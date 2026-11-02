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

void Header(const char* title) { std::printf("\n== %s ==\n", title); }

// One row per S(T): value and a bar of '#' (or '-' for negative values).
void Diagram(const char* name, const Portfolio& p, double lo, double hi,
             std::size_t n, bool pnl = false) {
  std::printf("%s\n   S(T)    %s\n", name, pnl ? "P&L" : "V(T)");
  const auto rows = PayoffTable(p, lo, hi, n);
  double vmax = 1e-12;
  for (const auto& [s, v0] : rows)
    vmax = std::max(vmax, std::abs(pnl ? v0 - p.Cost() : v0));
  const double scale = 30.0 / vmax;  // longest bar: 30 characters
  for (const auto& [s, v0] : rows) {
    const double v = pnl ? v0 - p.Cost() : v0;
    const auto len = static_cast<std::size_t>(std::lround(std::abs(v) * scale));
    std::printf("  %6.2f %7.2f  %s\n", s, v,
                std::string(len, v < 0 ? '-' : '#').c_str());
  }
}

}  // namespace

int main() {
  Header("Ex 1-6: antiderivatives (F(2) - F(1))");
  std::printf("  1  int_1^2 ln x dx            = %.10f\n",
              Ex01LnAntiderivative(2) - Ex01LnAntiderivative(1));
  std::printf("  2  int_2^3 dx/(x ln x)        = %.10f\n",
              Ex02InvXLnXAntiderivative(3) - Ex02InvXLnXAntiderivative(2));
  std::printf("  3  int_0^1 dx/(1+x^2)         = %.10f  (pi/4)\n",
              Ex03Inv1PlusX2Antiderivative(1));
  std::printf(
      "  4  int_1^2 x^3 ln x dx        = %.10f\n",
      Ex04XPowNLnAntiderivative(3, 2) - Ex04XPowNLnAntiderivative(3, 1));
  std::printf(
      "  5  int_1^2 x^3 e^x dx         = %.10f\n",
      Ex05XPowNExpAntiderivative(3, 2) - Ex05XPowNExpAntiderivative(3, 1));
  std::printf("  6  int_1^2 (ln x)^3 dx        = %.10f\n",
              Ex06LnPowNAntiderivative(3, 2) - Ex06LnPowNAntiderivative(3, 1));

  Header("Ex 7-10: limits");
  for (double x : {1.0, 10.0, 1000.0}) {
    const auto [lo, hi] = Ex07EBounds(x);
    std::printf("  7  x = %6.0f: %.8f < e < %.8f\n", x, lo, hi);
  }
  std::printf(
      "  8  sqrt quotient at 1e-6 = %.10f (limit -1/8); exp quotient = %.10f "
      "(limit 1/6)\n",
      Ex08SqrtQuotient(1e-6), Ex08ExpQuotient(1e-6));
  std::printf(
      "  9  (i) at 1e8 = %.10f (limit -1/2); (ii) at 1e8 = %.4e (-> -inf)\n",
      Ex09ExprI(1e8), Ex09ExprIi(1e8));
  std::printf(" 10  (1 - 1/x)^x at 1e8 = %.10f, 1/e = %.10f\n", Ex10Expr(1e8),
              std::exp(-1.0));

  Header("Ex 11: g'(x) for K = 100, T = 0.5, sigma = 0.3, r = 0.04");
  const Ex11Params p11{100, 0.5, 0.3, 0.04};
  for (double x : {80.0, 100.0, 120.0})
    std::printf("  x = %5.1f: g'(x) = %.8f\n", x, Ex11GPrime(p11, x));

  Header("Ex 12-13: limits of integrals");
  for (double h : {0.1, 0.01}) {
    std::printf(" 12  h = %.2f: average of cos on [1-h, 1+h] - cos(1) = %.3e\n",
                h,
                Ex12LocalAverage([](double x) { return std::cos(x); }, 1.0, h) -
                    std::cos(1.0));
  }
  for (double s : {0.1, 0.01}) {
    std::printf(
        " 13  sigma = %.2f: E[cos(0.5 + sigma Z)] - cos(0.5) = %.3e\n", s,
        Ex13GaussianAverage([](double x) { return std::cos(x); }, 0.5, s) -
            std::cos(0.5));
  }

  Header("Ex 14: 2-year 5% semiannual bond at y = 6%");
  std::vector<CashFlow> cfs{{0.5, 2.5}, {1.0, 2.5}, {1.5, 2.5}, {2.0, 102.5}};
  const auto f = Ex14Continuous(cfs, 0.06);
  const auto g = Ex14Discrete(cfs, 0.06, 2);
  std::printf("  continuous: f = %.6f, f' = %.6f, f'' = %.6f\n", f.value,
              f.first, f.second);
  std::printf("  semiannual: g = %.6f, g' = %.6f, g'' = %.6f\n", g.value,
              g.first, g.second);

  Header("Ex 15: f at a = (1, -1, 0)");
  const auto grad = Ex15Gradient(kEx15A);
  std::printf(
      "  f(a) = %.0f, Df(a) = (%.0f, %.0f, %.0f), D2f = "
      "[[4,-1,0],[-1,0,3],[0,3,-2]]\n",
      Ex15F(kEx15A), grad[0], grad[1], grad[2]);

  Header("Ex 16: u_t - u_xx at (x, t) = (0.7, 0.3)");
  std::printf("  %.3e\n", Ex16UT(0.7, 0.3) - Ex16UXx(0.7, 0.3));

  Header("Ex 17: C = P iff K = F");
  std::printf("  S = 50, T = 1, r = 5%%, q = 2%%: F = %.4f\n",
              Ex17EqualValueStrike(50, 1, 0.05, 0.02));

  Header("Ex 18-21: spreads");
  Diagram("18  butterfly 30/35/40", Ex18Butterfly(), 26, 44, 18);
  Diagram("19  bull call spread 30/35", Ex19BullSpread(), 26, 40, 14);
  Diagram("19  bear put spread 20/15", Ex19BearSpread(), 10, 24, 14);
  std::printf(
      "20  butterfly 45/50/55 costs %.2f, profitable for %.0f < S(T) < %.0f\n",
      Ex20Butterfly().Cost(), kEx20BreakevenLow, kEx20BreakevenHigh);
  Diagram("20  P&L", Ex20Butterfly(), 42, 58, 16, true);
  std::printf(
      "21  Portfolio 1 - Portfolio 2 payoff at S = X: %.1f (never negative)\n",
      Ex21Portfolio1(50).Payoff(50) - Ex21Portfolio2(50).Payoff(50));

  Header("Ex 22-24: Put-Call parity arbitrage");
  if (const auto a = Ex22Arbitrage()) {
    std::printf(
        " 22  calls %+.0f, puts %+.0f, shares %+.5f, cash %+.4f -> profit at T "
        "%.4f\n",
        a->calls, a->puts, a->shares, a->cash, a->profit);
  }
  std::printf(" 23  arbitrage: %s\n", Ex23Arbitrage() ? "yes" : "no");

  Header("Ex 25-27: strategies");
  std::printf(" 25  crossover S = %.3f\n", kEx25Crossover);
  for (double s : {30.0, 35.0, 39.0, 40.0, 42.5, 45.0})
    std::printf("     S(T) = %4.1f: stock %+7.1f%%, calls %+7.1f%%\n", s,
                100 * Ex25StockReturn(s), 100 * Ex25CallReturn(s));
  const auto p26 = Ex26Portfolio();
  std::printf(" 26  cash %.0f, shares %.0f", p26.cash, p26.shares);
  for (const auto& leg : p26.options)
    std::printf(", %+.0f calls K = %.0f", leg.quantity, leg.strike);
  std::printf("\n");
  Diagram("     V(T)", p26, 0, 60, 12);
  const auto p27 = Ex27RangeDigital(10, 40, 60, 0.5);
  std::printf(" 27  c = 10 on [40, 60], eps = 0.5:");
  for (const auto& leg : p27.options)
    std::printf(" %+.0f C(%.1f)", leg.quantity, leg.strike);
  std::printf("\n");

  Header("Ex 28-30: arbitrage");
  const auto x28 = Ex28Data();
  std::printf(" 28  convexity gap %.4f (< 0: arbitrage)\n",
              Ex29ConvexityGap(x28));
  if (const auto a = Ex29Arbitrage(x28)) {
    for (const auto& leg : a->options)
      std::printf("     %+.0f calls K = %.0f\n", leg.quantity, leg.strike);
    std::printf("     cash collected now %.2f\n", -a->Cost());
    Diagram("     payoff", *a, 95, 135, 8);
  }
  if (const auto a = Ex30Arbitrage()) {
    std::printf(
        " 30  futures %+.0f, index units %+.5f, cash lent %.2f -> profit at T "
        "%.2f\n",
        a->futures, a->units, a->cash, a->profit);
  }
}
