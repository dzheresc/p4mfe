// Prints the numerical answers and checks for the exercises of section 8.12.
//
//   ./build/cpp/ch08_multivariable_barriers/exercises/ch08_exercises_solutions

#include <cmath>
#include <cstdio>
#include <numbers>
#include <p4mfe/ch08/exercises/ex01_gamma_maximum.hpp>
#include <p4mfe/ch08/exercises/ex02_03_double_integrals.hpp>
#include <p4mfe/ch08/exercises/ex04_05_polar_areas.hpp>
#include <p4mfe/ch08/exercises/ex06_07_exp_inequalities.hpp>
#include <p4mfe/ch08/exercises/ex08_gronwall.hpp>
#include <p4mfe/ch08/exercises/ex09_10_heat_change_of_variables.hpp>
#include <p4mfe/ch08/exercises/ex11_barrier_rebate.hpp>
#include <p4mfe/ch08/exercises/ex12_13_asian_pdes.hpp>
#include <p4mfe/ch08/exercises/ex14_itm_puts.hpp>
#include <p4mfe/ch08/exercises/ex15_time_value.hpp>
#include <p4mfe/ch08/exercises/ex16_17_down_and_out_call.hpp>
#include <p4mfe/ch08/exercises/ex18_doc_greeks.hpp>
#include <p4mfe/ch08/exercises/ex19_20_early_exercise.hpp>
#include <p4mfe/ch08/exercises/ex21_implied_vol_skew.hpp>

using namespace p4mfe;
using namespace p4mfe::ch08::ex;

namespace {
void Header(const char* title) { std::printf("\n== %s ==\n", title); }
}  // namespace

int main() {
  const Ex01Market m1;
  Header("Ex 1: Gamma of a call, K = 50, T = 0.5, sigma = 30%, r = 5%");
  std::printf(
      "  maximum at S* = K exp(-(r + 3 sigma^2/2) T) = %.6f, Gamma(S*) = "
      "%.6f\n",
      Ex01Argmax(m1), Ex01MaxGamma(m1));
  for (double spot : {1.0, 20.0, 40.0, Ex01Argmax(m1), 50.0, 70.0, 200.0})
    std::printf("    S = %-10.4f Gamma = %.6e\n", spot, Ex01Gamma(spot, m1));

  Header("Ex 2: integral over the triangle of (x - y)/(x + y)");
  std::printf("  exact 0; direct Simpson %.2e; in (s, t) %.2e\n", Ex02Direct(),
              Ex02ChangeOfVariables());

  Header("Ex 3: integral of x^2 - 2y between y = (x + 1)^2 and y = 5x - 1");
  std::printf("  exact -7/4; Simpson %.12f\n", Ex03Numeric());

  Header("Ex 4-5: areas");
  std::printf("  disk R = 2: polar %.12f, pi R^2 = %.12f\n",
              Ex04DiskAreaPolar(2.0), 4.0 * std::numbers::pi);
  std::printf("  ellipse a = 3, b = 0.5: %.10f, pi a b = %.10f\n",
              Ex05EllipseArea(3.0, 0.5), Ex05Exact(3.0, 0.5));

  Header("Ex 6: e^pi vs pi^e");
  std::printf("  e^pi = %.6f > pi^e = %.6f\n", Ex06EPowPi(), Ex06PiPowE());

  Header("Ex 7: (e^x + e^y)/2 - e^((x+y)/2) = (e^(x/2) - e^(y/2))^2/2 >= 0");
  for (auto [x, y] :
       {std::pair{0.0, 1.0}, std::pair{-2.0, 3.0}, std::pair{1.0, 1.0}})
    std::printf("  x = %4.1f, y = %4.1f: gap %.6f\n", x, y, Ex07Gap(x, y));

  Header("Ex 8: Gronwall, v = 1 + sin t, M = 2, x in [0, 6]");
  const double bound = 2.0;
  const auto c = Ex08ComputeCheck(
      [&](double x) { return bound * std::exp(0.5 * Ex08Integral(x)); },
      Ex08Integrand, bound, 6.0);
  std::printf(
      "  u = M e^(V/2): max(u - W) = %.3e, max(u/(M e^V) - 1) = %.3e, max g "
      "increase = %.3e\n",
      c.hypothesis_gap, c.conclusion_gap, c.g_increase);

  Header("Ex 9-10: heat equation variables, sigma = 30%, r = 5%, q = 2%");
  const auto ab = Ex09ComputeCoefficients(0.3, 0.05, 0.02);
  std::printf(
      "  a = %.6f, b = %.6f; printed b of ex 10 = %.6f (off by 2q/sigma^2 = "
      "%.6f)\n",
      ab.a, ab.b, Ex10PrintedB(0.3, 0.05, 0.02), 2.0 * 0.02 / 0.09);
  std::printf("  u_tau - u_xx at x = 0, tau = 0.05: call %.2e, put %.2e\n",
              Ex09HeatResidual(ch03::OptionType::kCall, 0.0, 0.05, 50.0, 0.3,
                               0.05, 0.02),
              Ex09HeatResidual(ch03::OptionType::kPut, 0.0, 0.05, 50.0, 0.3,
                               0.05, 0.02));

  Header(
      "Ex 11: rebate R = 2 at B = 35 (S = K = 40, T = 0.5, sigma = 30%, r = "
      "5%)");
  std::printf(
      "  u(ln(B/K), tau) = R (B/K)^a e^(b tau): %.6f at tau = 0, %.6f at tau = "
      "sigma^2 T/2\n",
      Ex11UBoundary(0.0, 35.0, 40.0, 2.0, 0.3, 0.05),
      Ex11UBoundary(0.0225, 35.0, 40.0, 2.0, 0.3, 0.05));
  std::printf("  heat FD with this boundary: %.6f, closed form: %.6f\n",
              Ex11HeatFd(40, 40, 35, 2, 0.5, 0.3, 0.05),
              Ex11DownAndOutCallWithRebate(40, 40, 35, 2, 0.5, 0.3, 0.05));

  Header("Ex 12-13: Asian option PDEs");
  const AsianMarket am;
  const auto geo_price = [&](double y, double t) {
    return Ex12GeometricAsianCall(y, t, am);
  };
  std::printf(
      "  geometric Asian call F(ln 50, 0.3) = %.6f, F-PDE residual %.2e\n",
      geo_price(std::log(50.0), 0.3),
      Ex12FResidual(geo_price, std::log(50.0), 0.3, am.maturity, am.sigma,
                    am.r));
  const auto linear_h = [&](double ratio, double t) {
    return Ex13HLinear(ratio, t, 1.0, 0.05);
  };
  std::printf("  H = alpha(t) + beta(t) R: (8.55) residual %.2e\n",
              Ex13HResidual(linear_h, 0.4, 0.5, 0.3, 0.05));

  Header("Ex 14: ITM puts, S = 40, sigma = 30%, r = 5%");
  std::printf("  Strike  Maturity  BS value   Intrinsic  Intrinsic > BS\n");
  for (const auto& row : Ex14Table())
    std::printf("  %5.0f   %4.0f mo   %9.6f  %5.0f      %s\n", row.strike,
                row.maturity * 12, row.value, row.intrinsic,
                row.intrinsic_larger ? "yes" : "no");

  Header("Ex 15: time value C - (S - K)^+, K = 40");
  for (double spot : {30.0, 38.0, 40.0, 42.0, 50.0})
    std::printf("  S = %4.0f: %.6f\n", spot, Ex15TimeValue(spot));

  Header(
      "Ex 16: down-and-out call, S = K = 40, B = 35, T = 0.5, sigma = 30%, r = "
      "5%");
  std::printf("  V_DOC = %.6f (vanilla %.6f)\n", Ex16Formula846(),
              ch03::BsCall(40, 40, 0.5, 0.3, 0.05));

  Header("Ex 17: B -> 0, r = q = 0 (S = K = 40, T = 2, sigma = 60%)");
  for (double barrier : {35.0, 20.0, 10.0, 5.0, 1.0})
    std::printf("  B = %4.0f: C - V = %.6e, N(d2)/B = %.6e\n", barrier,
                Ex17Bracket(40, 40, barrier, 2, 0.6),
                Ex17ND2OverB(40, 40, barrier, 2, 0.6));

  Header("Ex 18: Delta and Gamma of the down-and-out call of ex 16");
  for (double spot : {36.0, 40.0, 50.0}) {
    const auto g = Ex18Greeks(spot, 40, 35, 0.5, 0.3, 0.05);
    const auto f = Ex18GreeksFd(spot, 40, 35, 0.5, 0.3, 0.05);
    std::printf("  S = %2.0f: Delta = %.6f (FD %.6f), Gamma = %.6f (FD %.6f)\n",
                spot, g.delta, f.delta, g.gamma, f.gamma);
  }

  Header(
      "Ex 19: European = American call for q = 0 (K = 40, T = 1, sigma = 30%, "
      "r = 5%)");
  for (double spot : {30.0, 40.0, 60.0})
    std::printf(
        "  S = %2.0f: time value %.6f, binomial early exercise premium %.2e\n",
        spot, Ex19TimeValue(spot),
        ch08::EarlyExercisePremium(ch03::OptionType::kCall, spot, 40, 1, 0.3,
                                   0.05, 0.0));

  Header(
      "Ex 20: with dividends C < S - K for large S (K = 40, T = 1, sigma = "
      "30%, r = 5%)");
  for (double q : {0.05, 0.01, 0.001}) {
    const CallMarket cm{40.0, 1.0, 0.3, 0.05, q};
    std::printf(
        "  q = %5.3f: C < S - K for S > S* = %.4f (sufficient: S >= K/(1 - "
        "e^(-qT)) = %.2f)\n",
        q, Ex20Crossover(cm), Ex20Threshold(cm));
  }

  Header("Ex 21: d sigma_imp/dK from prices, smile 0.2 - 0.1 k + 0.15 k^2");
  for (double strike : {80.0, 100.0, 120.0})
    std::printf(
        "  K = %3.0f: from prices %.8f, exact %.8f\n", strike,
        Ex21SlopeFromPrices([](double k) { return Ex21Price(k); }, strike),
        Ex21SmileSlope(strike));
  return 0;
}
