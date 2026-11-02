// Prints the numerical answers to the exercises of section 3.11.
//
//   ./build/cpp/ch03_probability_black_scholes/exercises/ch03_exercises_solutions

#include <cmath>
#include <cstdio>
#include <p4mfe/ch03/exercises/ex01_04_dice_coins.hpp>
#include <p4mfe/ch03/exercises/ex05_09_distributions.hpp>
#include <p4mfe/ch03/exercises/ex10_11_cauchy_schwarz.hpp>
#include <p4mfe/ch03/exercises/ex12_15_black_scholes.hpp>
#include <p4mfe/ch03/exercises/ex16_21_greeks_strike.hpp>
#include <p4mfe/ch03/exercises/ex22_24_maturity_bounds.hpp>
#include <p4mfe/ch03/exercises/ex25_29_hedging.hpp>
#include <vector>

using namespace p4mfe::ch03;
using namespace p4mfe::ch03::ex;

namespace {
void header(const char* title) { std::printf("\n== %s ==\n", title); }
}  // namespace

int main() {
  header("Ex 1: smallest w(k) that makes the dice game worth playing");
  for (int k = 2; k <= 12; ++k)
    std::printf("  k = %2d  p = %2d/36  w(k) = %.4g\n", k, ex01_ways(k),
                ex01_breakeven_win(k));

  header("Ex 2: flips until the first head");
  std::printf("  E[X] = %g  var(X) = %g\n", ex02_mean(), ex02_variance());

  header(
      "Ex 3-4: tosses until k heads in a row, E_k = (1 - p^k) / ((1 - p) p^k)");
  std::printf(
      "  fair coin, k = 2: %g;  biased: (1 + p)/p^2, e.g. p = 0.6: %.6f\n",
      ex03_expected_tosses(), ex03_expected_tosses(0.6));
  for (int k = 1; k <= 5; ++k)
    std::printf("  k = %d  p = 0.5: %6.1f  p = 0.3: %10.4f\n", k,
                ex04_expected_tosses(k, 0.5), ex04_expected_tosses(k, 0.3));

  header("Ex 5: three-step binomial stock");
  const auto d5 = ex05_distribution();
  for (std::size_t i = 0; i < d5.values.size(); ++i)
    std::printf("  S_T = %.5f  prob %.3f\n", d5.values[i], d5.probs[i]);
  std::printf("  E[S_T] = %.6f  var(S_T) = %.6f  (sd %.4f)\n", ex05_mean(),
              ex05_variance(), std::sqrt(ex05_variance()));

  header("Ex 6-9: moments");
  std::printf("  uniform [2, 5]: mean %g  var %g\n", ex06_mean(2, 5),
              ex06_variance(2, 5));
  const auto e7 = exponential_distribution(2.0);
  std::printf(
      "  exponential alpha = 2: mean %.8f (1/2)  var %.8f (1/4)  P(X >= 1) = "
      "%.8f\n",
      mean(e7), variance(e7), ex07_tail(2.0, 1.0));
  std::printf("  normal mu = 1, sigma = 2: E|X| = %.8f  E[X^2] = %g\n",
              ex08_mean_abs(1, 2), ex08_second_moment(1, 2));
  const auto d9 = ex09_distribution(3.5);
  std::printf("  Poisson lambda = 3.5: mean %.12f  var %.12f\n", mean(d9),
              variance(d9));

  header("Ex 10-11: Cauchy-Schwarz");
  const std::vector<double> x{1, 2, 3}, y{4, -5, 6};
  const auto s10 = ex10_sides(x, y);
  std::printf("  sums: (x.y)^2 = %g <= %g\n", s10.lhs, s10.rhs);
  const auto s11 = ex11_sides([](double t) { return std::exp(t); },
                              [](double t) { return std::sin(t); }, 0.0, 2.0);
  std::printf("  int e^x sin x on [0,2] = %.8f <= %.8f\n", s11.lhs, s11.rhs);

  header("Ex 12: S = 50, K = 45, T = 0.5, sigma = 0.2, q = 0.02, r = 0.06");
  const auto [d1, d2] = bs_d1_d2(50, 45, 0.5, 0.2, 0.06, 0.02);
  std::printf("  d1 = %.6f  d2 = %.6f  C = %.6f  P = %.6f\n", d1, d2,
              ex12_call(), ex12_put());

  header("Ex 13: zero strike");
  std::printf(
      "  P = %g,  C = S e^(-qT): S = 80, q = 0.03, T = 2 -> %.6f (BS with K = "
      "1e-8: %.6f)\n",
      ex13_put_zero_strike(), ex13_call_zero_strike(80, 2, 0.03),
      bs_call(80, 1e-8, 2, 0.3, 0.05, 0.03));

  header("Ex 14-15");
  std::printf("  rho(P) via parity %.8f  vs library %.8f\n",
              ex14_rho_put_from_parity(50, 45, 0.5, 0.2, 0.06, 0.02),
              bs_greeks(OptionType::kPut, 50, 45, 0.5, 0.2, 0.06, 0.02).rho);
  std::printf(
      "  AoN call Delta (S = K = 40, T = 0.5, sigma = 0.3, r = 0.05, q = "
      "0.01): %.6f\n",
      ex15_aon_delta(40, 40, 0.5, 0.3, 0.05, 0.01));

  header(
      "Ex 16: volga and vanna (S = 50, K = 45, T = 0.5, sigma = 0.2, r = 0.06, "
      "q = 0.02)");
  std::printf("  volga = %.6f  vanna = %.6f  (same for call and put)\n",
              ex16_volga(50, 45, 0.5, 0.2, 0.06, 0.02),
              ex16_vanna(50, 45, 0.5, 0.2, 0.06, 0.02));

  header("Ex 17: ATM C - P = S (e^(-qT) - e^(-rT))");
  for (double q : {0.02, 0.05, 0.08})
    std::printf("  r = 0.05, q = %.2f: C - P = %+.6f\n", q,
                ex17_atm_call_minus_put(100, 1, 0.2, 0.05, q));

  header("Ex 18: ATM call Theta with q = r = 0.05, sigma = 0.2, S = 100");
  for (double T : {1.0, 5.0, 9.0, 10.0, 20.0})
    std::printf("  T = %4.1f  Theta = %+.6f\n", T,
                ex18_theta_call(100, 100, T, 0.2, 0.05, 0.05));
  std::printf("  sign change at T* = %.4f years\n",
              ex18_theta_sign_change(0.2, 0.05));

  header("Ex 19-21 (S = 50, K = 45, T = 0.5, sigma = 0.2, r = 0.06, q = 0.02)");
  const auto sd = ex19_strike_derivatives(50, 45, 0.5, 0.2, 0.06, 0.02);
  std::printf("  dC/dK = %.6f  d2C/dK2 = %.6f  dP/dK = %.6f  d2P/dK2 = %.6f\n",
              sd.dC_dK, sd.d2C_dK2, sd.dP_dK, sd.d2P_dK2);
  const double Ks = ex21_min_straddle_strike(50, 0.5, 0.2, 0.06, 0.02);
  std::printf(
      "  min straddle strike K* = S e^((r - q - sigma^2/2) T) = %.6f, straddle "
      "%.6f\n",
      Ks, ex21_straddle(50, Ks, 0.5, 0.2, 0.06, 0.02));

  header("Ex 22-23: ATM Gamma (r = 5%) and vega (r = 0), S = 50, sigma = 30%");
  const char* names[] = {"15 days ", "3 months", "1 year  "};
  for (int i = 0; i < 3; ++i)
    std::printf("  %s  Gamma = %.6f  vega = %.6f\n", names[i],
                ex22_atm_gamma(ex22_maturities[i]),
                ex23_atm_vega(ex22_maturities[i]));
  std::printf("  d(vega)/d(tau) > 0 for tau < 4/sigma^2 = %.2f years\n",
              4 / 0.09);

  header("Ex 24: bounds, S = 100, K = 100, T = 1, r = 0.05, q = 0.02");
  const auto cb = ex24_call_bounds(100, 100, 1, 0.05, 0.02);
  const auto pb = ex24_put_bounds(100, 100, 1, 0.05, 0.02);
  std::printf("  %.4f <= C = %.4f <= %.4f;  %.4f <= P = %.4f <= %.4f\n",
              cb.lower, bs_call(100, 100, 1, 0.25, 0.05, 0.02), cb.upper,
              pb.lower, bs_put(100, 100, 1, 0.25, 0.05, 0.02), pb.upper);

  header("Ex 25: Delta 5000, Gamma -200");
  const auto h25 = ex25_options_only(5000, -200, 0.4, 0.05, -0.5, 0.07);
  std::printf("  options only: %.2f calls, %.2f puts\n", h25.calls, h25.puts);
  const auto c25 = delta_gamma_hedge(5000, -200, 0.4, 0.05);
  const auto p25 = delta_gamma_hedge(5000, -200, -0.5, 0.07);
  std::printf(
      "  with the asset: %.2f calls and %.2f shares, or %.2f puts and %.2f "
      "shares\n",
      c25.hedge_options, c25.shares, p25.hedge_options, p25.shares);

  header("Ex 26");
  std::printf("  shares = %.4f\n", ex26_shares());

  header("Ex 27");
  const auto e27 = ex27();
  std::printf("  (i) premium %.2f  (ii) shares %.4f, cash %.2f\n", e27.premium,
              e27.shares, e27.cash);
  std::printf(
      "  (iii) options + shares %.2f; with cash %.4f  (iv) unhedged loss "
      "%.2f\n",
      e27.position_value, e27.portfolio_value, e27.unhedged_loss);

  header("Ex 28");
  const auto e28 = ex28();
  std::printf("  (i) %.2f  (ii) Delta %.4f -> hold %.4f shares, cash %.2f\n",
              e28.value0, e28.port_delta0, e28.shares1, e28.cash1);
  std::printf("  (iii) value %.2f, hold %.4f shares (sell %.4f)\n", e28.value1,
              e28.shares2, e28.shares1 - e28.shares2);

  header("Ex 29");
  const auto h29 = ex29_hedge();
  std::printf("  calls %.2f  puts %.2f  shares %.2f\n", h29.calls, h29.puts,
              h29.shares);
}
