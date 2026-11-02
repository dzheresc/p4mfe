// Prints the numerical answers to the exercises of section 4.9.
//
//   ./build/cpp/ch04_lognormal_risk_neutral/exercises/ch04_exercises_solutions

#include <cmath>
#include <cstdio>
#include <p4mfe/ch04/exercises/ex01_02_normal_averages.hpp>
#include <p4mfe/ch04/exercises/ex03_06_binomial_calibration.hpp>
#include <p4mfe/ch04/exercises/ex07_08_confidence_roulette.hpp>
#include <p4mfe/ch04/exercises/ex09_10_itm_delta.hpp>
#include <p4mfe/ch04/exercises/ex11_put_integration.hpp>
#include <p4mfe/ch04/exercises/ex12_13_power_payoffs.hpp>
#include <p4mfe/ch04/exercises/ex14_15_normal_model_inverse.hpp>

using namespace p4mfe;
using namespace p4mfe::ch04;
using namespace p4mfe::ch04::ex;

namespace {
void Header(const char* title) { std::printf("\n== %s ==\n", title); }
}  // namespace

int main() {
  Header("Ex 1-2: sums and averages of normals");
  const auto s1 = Ex01SumIndependent();
  std::printf(
      "  X1 + X2 (independent N(0,1)): mean %g  variance %g;  as written (X2 = "
      "-X1): %g\n",
      s1.mean, s1.variance, Ex01SumAsWritten(0.7));
  for (std::size_t n : {1u, 10u, 100u}) {
    const auto a = Ex02Average(5.0, 2.0, n);
    std::printf(
        "  average of n = %3zu, mu = 5, sigma = 2: mean %g  variance %g\n", n,
        a.mean, a.variance);
  }

  Header("Ex 5-6: binomial trees calibrated to r = 5%, sigma = 25%");
  for (double dt : {1.0 / 252, 1.0 / 12}) {
    const auto c = Ex05Crr(0.05, 0.25, dt);
    const auto e = Ex06EqualProbability(0.05, 0.25, dt);
    std::printf(
        "  dt = %.5f  CRR: u = %.6f d = %.6f p = %.6f   p = 1/2: u = %.6f d = "
        "%.6f\n",
        dt, c.u, c.d, c.p, e.u, e.d);
  }

  Header("Ex 7: confidence intervals, S0 = 50, q = 2%, sigma = 30%");
  const char* labels[] = {"15 days", "1 month", "2 months", "6 months",
                          "1 year"};
  for (bool rn : {false, true}) {
    std::printf("  %s\n",
                rn ? "(ii) risk-neutral, mu = r = 5%" : "(i) mu = 8%");
    for (std::size_t i = 0; i < kEx07Horizons.size(); ++i) {
      const auto a = Ex07Interval(0.95, kEx07Horizons[i], rn);
      const auto b = Ex07Interval(0.99, kEx07Horizons[i], rn);
      std::printf("    %-8s 95%% [%8.4f, %8.4f]  99%% [%8.4f, %8.4f]\n",
                  labels[i], a.lo, a.hi, b.lo, b.hi);
    }
  }

  Header("Ex 8: 100 bets of $100 on black (American roulette)");
  const auto e8 = Ex08Exact(), n8 = Ex08Normal(), c8 = Ex08Normal(true);
  std::printf(
      "  exact:                      P(win >= 1000) = %.6f  P(lose >= 1000) = "
      "%.6f\n",
      e8.win, e8.lose);
  std::printf(
      "  normal approximation:       P(win >= 1000) = %.6f  P(lose >= 1000) = "
      "%.6f\n",
      n8.win, n8.lose);
  std::printf(
      "  with continuity correction: P(win >= 1000) = %.6f  P(lose >= 1000) = "
      "%.6f\n",
      c8.win, c8.lose);

  Header(
      "Ex 9: put K = 55, T = 4 months, S = 60, mu = 10%, sigma = 30%, r = 5%");
  std::printf("  P(ITM) = %.6f  risk-neutral P(ITM) = %.6f  N(-d2) = %.6f\n",
              Ex09ProbItm(), Ex09RnProbItm(), Ex09NMinusD2());

  Header(
      "Ex 10: dividend yield q* above which Delta(ATM call) < 0.5 (r = 5%, "
      "sigma = 20%)");
  for (double maturity : {0.25, 1.0, 5.0})
    std::printf("  T = %4.2f  Delta(q = 0) = %.6f  q* = %.6f  (approx %.6f)\n",
                maturity, Ex10AtmCallDelta(0.05, 0.0, 0.2, maturity),
                Ex10ThresholdQ(0.05, 0.2, maturity),
                Ex10ThresholdQApprox(0.05, 0.2, maturity));

  Header(
      "Ex 11: put K = 50, T = 0.5, q = 2%, r = 4% (assumed S0 = 50, sigma = "
      "30%)");
  std::printf("  P_BS = %.9f\n", Ex11PutBs());
  for (auto rule : {ch02::Rule::kMidpoint, ch02::Rule::kSimpson}) {
    std::printf("  %s rule\n",
                rule == ch02::Rule::kMidpoint ? "Midpoint" : "Simpson's");
    for (const auto& s : Ex11Run(rule))
      std::printf("    n = %4zu  %.9f  error % .3e\n", s.n, s.value, s.error);
  }

  Header(
      "Ex 12-13: squared call and power call, S0 = 50, K = 50, T = 1, sigma = "
      "30%, r = 5%");
  std::printf("  ((S - K)^+)^2: %.6f\n", Ex12SquaredCall(50, 50, 1, 0.3, 0.05));
  for (double alpha : {0.5, 1.0, 2.0})
    std::printf("  (S^%.1f - K)^+ with K = 50^%.1f: %.6f\n", alpha, alpha,
                Ex13PowerCall(50, std::pow(50.0, alpha), 1, 0.3, 0.05, alpha));

  Header(
      "Ex 14: normal model, S0 = 50, K = 52, T = 0.75, sigma = 8 (price "
      "units), r = 4%");
  std::printf(
      "  C = %.6f  P (parity) = %.6f  P (direct, book's mean S0 + rT) = %.6f\n",
      Ex14Call(50, 52, 0.75, 8, 0.04), Ex14PutParity(50, 52, 0.75, 8, 0.04),
      Ex14PutForward(Ex14BookForward(50, 0.04, 0.75), 52, 0.75, 8, 0.04));

  Header("Ex 15: S0 = 50, K = 50, T = 1, sigma = 30%, r = 5%, q = 2%");
  std::printf("  pays 1/S(T): %.8f   pays (1/K - 1/S(T))^+: %.8f\n",
              Ex15InverseClaim(50, 1, 0.3, 0.05, 0.02),
              Ex15InverseCall(50, 50, 1, 0.3, 0.05, 0.02));
}
