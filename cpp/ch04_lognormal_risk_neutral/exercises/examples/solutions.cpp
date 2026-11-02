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
void header(const char* title) { std::printf("\n== %s ==\n", title); }
}  // namespace

int main() {
  header("Ex 1-2: sums and averages of normals");
  const auto s1 = ex01_sum_independent();
  std::printf(
      "  X1 + X2 (independent N(0,1)): mean %g  variance %g;  as written (X2 = "
      "-X1): %g\n",
      s1.mean, s1.variance, ex01_sum_as_written(0.7));
  for (std::size_t n : {1u, 10u, 100u}) {
    const auto a = ex02_average(5.0, 2.0, n);
    std::printf(
        "  average of n = %3zu, mu = 5, sigma = 2: mean %g  variance %g\n", n,
        a.mean, a.variance);
  }

  header("Ex 5-6: binomial trees calibrated to r = 5%, sigma = 25%");
  for (double dt : {1.0 / 252, 1.0 / 12}) {
    const auto c = ex05_crr(0.05, 0.25, dt);
    const auto e = ex06_equal_probability(0.05, 0.25, dt);
    std::printf(
        "  dt = %.5f  CRR: u = %.6f d = %.6f p = %.6f   p = 1/2: u = %.6f d = "
        "%.6f\n",
        dt, c.u, c.d, c.p, e.u, e.d);
  }

  header("Ex 7: confidence intervals, S0 = 50, q = 2%, sigma = 30%");
  const char* labels[] = {"15 days", "1 month", "2 months", "6 months",
                          "1 year"};
  for (bool rn : {false, true}) {
    std::printf("  %s\n",
                rn ? "(ii) risk-neutral, mu = r = 5%" : "(i) mu = 8%");
    for (std::size_t i = 0; i < ex07_horizons.size(); ++i) {
      const auto a = ex07_interval(0.95, ex07_horizons[i], rn);
      const auto b = ex07_interval(0.99, ex07_horizons[i], rn);
      std::printf("    %-8s 95%% [%8.4f, %8.4f]  99%% [%8.4f, %8.4f]\n",
                  labels[i], a.lo, a.hi, b.lo, b.hi);
    }
  }

  header("Ex 8: 100 bets of $100 on black (American roulette)");
  const auto e8 = ex08_exact(), n8 = ex08_normal(), c8 = ex08_normal(true);
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

  header(
      "Ex 9: put K = 55, T = 4 months, S = 60, mu = 10%, sigma = 30%, r = 5%");
  std::printf("  P(ITM) = %.6f  risk-neutral P(ITM) = %.6f  N(-d2) = %.6f\n",
              ex09_prob_itm(), ex09_rn_prob_itm(), ex09_n_minus_d2());

  header(
      "Ex 10: dividend yield q* above which Delta(ATM call) < 0.5 (r = 5%, "
      "sigma = 20%)");
  for (double T : {0.25, 1.0, 5.0})
    std::printf("  T = %4.2f  Delta(q = 0) = %.6f  q* = %.6f  (approx %.6f)\n",
                T, ex10_atm_call_delta(0.05, 0.0, 0.2, T),
                ex10_threshold_q(0.05, 0.2, T),
                ex10_threshold_q_approx(0.05, 0.2, T));

  header(
      "Ex 11: put K = 50, T = 0.5, q = 2%, r = 4% (assumed S0 = 50, sigma = "
      "30%)");
  std::printf("  P_BS = %.9f\n", ex11_put_bs());
  for (auto rule : {ch02::Rule::Midpoint, ch02::Rule::Simpson}) {
    std::printf("  %s rule\n",
                rule == ch02::Rule::Midpoint ? "Midpoint" : "Simpson's");
    for (const auto& s : ex11_run(rule))
      std::printf("    n = %4zu  %.9f  error % .3e\n", s.n, s.value, s.error);
  }

  header(
      "Ex 12-13: squared call and power call, S0 = 50, K = 50, T = 1, sigma = "
      "30%, r = 5%");
  std::printf("  ((S - K)^+)^2: %.6f\n",
              ex12_squared_call(50, 50, 1, 0.3, 0.05));
  for (double alpha : {0.5, 1.0, 2.0})
    std::printf(
        "  (S^%.1f - K)^+ with K = 50^%.1f: %.6f\n", alpha, alpha,
        ex13_power_call(50, std::pow(50.0, alpha), 1, 0.3, 0.05, alpha));

  header(
      "Ex 14: normal model, S0 = 50, K = 52, T = 0.75, sigma = 8 (price "
      "units), r = 4%");
  std::printf(
      "  C = %.6f  P (parity) = %.6f  P (direct, book's mean S0 + rT) = %.6f\n",
      ex14_call(50, 52, 0.75, 8, 0.04), ex14_put_parity(50, 52, 0.75, 8, 0.04),
      ex14_put_forward(ex14_book_forward(50, 0.04, 0.75), 52, 0.75, 8, 0.04));

  header("Ex 15: S0 = 50, K = 50, T = 1, sigma = 30%, r = 5%, q = 2%");
  std::printf("  pays 1/S(T): %.8f   pays (1/K - 1/S(T))^+: %.8f\n",
              ex15_inverse_claim(50, 1, 0.3, 0.05, 0.02),
              ex15_inverse_call(50, 50, 1, 0.3, 0.05, 0.02));
}
