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
void Header(const char* title) { std::printf("\n== %s ==\n", title); }
}  // namespace

int main() {
  Header("Ex 1: smallest w(k) that makes the dice game worth playing");
  for (int k = 2; k <= 12; ++k)
    std::printf("  k = %2d  p = %2d/36  w(k) = %.4g\n", k, Ex01Ways(k),
                Ex01BreakevenWin(k));

  Header("Ex 2: flips until the first head");
  std::printf("  E[X] = %g  var(X) = %g\n", Ex02Mean(), Ex02Variance());

  Header(
      "Ex 3-4: tosses until k heads in a row, E_k = (1 - p^k) / ((1 - p) p^k)");
  std::printf(
      "  fair coin, k = 2: %g;  biased: (1 + p)/p^2, e.g. p = 0.6: %.6f\n",
      Ex03ExpectedTosses(), Ex03ExpectedTosses(0.6));
  for (int k = 1; k <= 5; ++k)
    std::printf("  k = %d  p = 0.5: %6.1f  p = 0.3: %10.4f\n", k,
                Ex04ExpectedTosses(k, 0.5), Ex04ExpectedTosses(k, 0.3));

  Header("Ex 5: three-step binomial stock");
  const auto d5 = Ex05Distribution();
  for (std::size_t i = 0; i < d5.values.size(); ++i)
    std::printf("  S_T = %.5f  prob %.3f\n", d5.values[i], d5.probs[i]);
  std::printf("  E[S_T] = %.6f  var(S_T) = %.6f  (sd %.4f)\n", Ex05Mean(),
              Ex05Variance(), std::sqrt(Ex05Variance()));

  Header("Ex 6-9: moments");
  std::printf("  uniform [2, 5]: mean %g  var %g\n", Ex06Mean(2, 5),
              Ex06Variance(2, 5));
  const auto e7 = ExponentialDistribution(2.0);
  std::printf(
      "  exponential alpha = 2: mean %.8f (1/2)  var %.8f (1/4)  P(X >= 1) = "
      "%.8f\n",
      Mean(e7), Variance(e7), Ex07Tail(2.0, 1.0));
  std::printf("  normal mu = 1, sigma = 2: E|X| = %.8f  E[X^2] = %g\n",
              Ex08MeanAbs(1, 2), Ex08SecondMoment(1, 2));
  const auto d9 = Ex09Distribution(3.5);
  std::printf("  Poisson lambda = 3.5: mean %.12f  var %.12f\n", Mean(d9),
              Variance(d9));

  Header("Ex 10-11: Cauchy-Schwarz");
  const std::vector<double> x{1, 2, 3}, y{4, -5, 6};
  const auto s10 = Ex10Sides(x, y);
  std::printf("  sums: (x.y)^2 = %g <= %g\n", s10.lhs, s10.rhs);
  const auto s11 = Ex11Sides([](double t) { return std::exp(t); },
                             [](double t) { return std::sin(t); }, 0.0, 2.0);
  std::printf("  int e^x sin x on [0,2] = %.8f <= %.8f\n", s11.lhs, s11.rhs);

  Header("Ex 12: S = 50, K = 45, T = 0.5, sigma = 0.2, q = 0.02, r = 0.06");
  const auto [d1, d2] = BsD1D2(50, 45, 0.5, 0.2, 0.06, 0.02);
  std::printf("  d1 = %.6f  d2 = %.6f  C = %.6f  P = %.6f\n", d1, d2,
              Ex12Call(), Ex12Put());

  Header("Ex 13: zero strike");
  std::printf(
      "  P = %g,  C = S e^(-qT): S = 80, q = 0.03, T = 2 -> %.6f (BS with K = "
      "1e-8: %.6f)\n",
      Ex13PutZeroStrike(), Ex13CallZeroStrike(80, 2, 0.03),
      BsCall(80, 1e-8, 2, 0.3, 0.05, 0.03));

  Header("Ex 14-15");
  std::printf("  rho(P) via parity %.8f  vs library %.8f\n",
              Ex14RhoPutFromParity(50, 45, 0.5, 0.2, 0.06, 0.02),
              BsGreeks(OptionType::kPut, 50, 45, 0.5, 0.2, 0.06, 0.02).rho);
  std::printf(
      "  AoN call Delta (S = K = 40, T = 0.5, sigma = 0.3, r = 0.05, q = "
      "0.01): %.6f\n",
      Ex15AonDelta(40, 40, 0.5, 0.3, 0.05, 0.01));

  Header(
      "Ex 16: volga and vanna (S = 50, K = 45, T = 0.5, sigma = 0.2, r = 0.06, "
      "q = 0.02)");
  std::printf("  volga = %.6f  vanna = %.6f  (same for call and put)\n",
              Ex16Volga(50, 45, 0.5, 0.2, 0.06, 0.02),
              Ex16Vanna(50, 45, 0.5, 0.2, 0.06, 0.02));

  Header("Ex 17: ATM C - P = S (e^(-qT) - e^(-rT))");
  for (double q : {0.02, 0.05, 0.08})
    std::printf("  r = 0.05, q = %.2f: C - P = %+.6f\n", q,
                Ex17AtmCallMinusPut(100, 1, 0.2, 0.05, q));

  Header("Ex 18: ATM call Theta with q = r = 0.05, sigma = 0.2, S = 100");
  for (double maturity : {1.0, 5.0, 9.0, 10.0, 20.0})
    std::printf("  T = %4.1f  Theta = %+.6f\n", maturity,
                Ex18ThetaCall(100, 100, maturity, 0.2, 0.05, 0.05));
  std::printf("  sign change at T* = %.4f years\n",
              Ex18ThetaSignChange(0.2, 0.05));

  Header("Ex 19-21 (S = 50, K = 45, T = 0.5, sigma = 0.2, r = 0.06, q = 0.02)");
  const auto sd = Ex19StrikeDerivatives(50, 45, 0.5, 0.2, 0.06, 0.02);
  std::printf("  dC/dK = %.6f  d2C/dK2 = %.6f  dP/dK = %.6f  d2P/dK2 = %.6f\n",
              sd.d_call_d_strike, sd.d2_call_d_strike2, sd.d_put_d_strike,
              sd.d2_put_d_strike2);
  const double min_strike = Ex21MinStraddleStrike(50, 0.5, 0.2, 0.06, 0.02);
  std::printf(
      "  min straddle strike K* = S e^((r - q - sigma^2/2) T) = %.6f, straddle "
      "%.6f\n",
      min_strike, Ex21Straddle(50, min_strike, 0.5, 0.2, 0.06, 0.02));

  Header("Ex 22-23: ATM Gamma (r = 5%) and vega (r = 0), S = 50, sigma = 30%");
  const char* names[] = {"15 days ", "3 months", "1 year  "};
  for (int i = 0; i < 3; ++i)
    std::printf("  %s  Gamma = %.6f  vega = %.6f\n", names[i],
                Ex22AtmGamma(kEx22Maturities[i]),
                Ex23AtmVega(kEx22Maturities[i]));
  std::printf("  d(vega)/d(tau) > 0 for tau < 4/sigma^2 = %.2f years\n",
              4 / 0.09);

  Header("Ex 24: bounds, S = 100, K = 100, T = 1, r = 0.05, q = 0.02");
  const auto cb = Ex24CallBounds(100, 100, 1, 0.05, 0.02);
  const auto pb = Ex24PutBounds(100, 100, 1, 0.05, 0.02);
  std::printf("  %.4f <= C = %.4f <= %.4f;  %.4f <= P = %.4f <= %.4f\n",
              cb.lower, BsCall(100, 100, 1, 0.25, 0.05, 0.02), cb.upper,
              pb.lower, BsPut(100, 100, 1, 0.25, 0.05, 0.02), pb.upper);

  Header("Ex 25: Delta 5000, Gamma -200");
  const auto h25 = Ex25OptionsOnly(5000, -200, 0.4, 0.05, -0.5, 0.07);
  std::printf("  options only: %.2f calls, %.2f puts\n", h25.calls, h25.puts);
  const auto c25 = ComputeDeltaGammaHedge(5000, -200, 0.4, 0.05);
  const auto p25 = ComputeDeltaGammaHedge(5000, -200, -0.5, 0.07);
  std::printf(
      "  with the asset: %.2f calls and %.2f shares, or %.2f puts and %.2f "
      "shares\n",
      c25.hedge_options, c25.shares, p25.hedge_options, p25.shares);

  Header("Ex 26");
  std::printf("  shares = %.4f\n", Ex26Shares());

  Header("Ex 27");
  const auto e27 = Ex27();
  std::printf("  (i) premium %.2f  (ii) shares %.4f, cash %.2f\n", e27.premium,
              e27.shares, e27.cash);
  std::printf(
      "  (iii) options + shares %.2f; with cash %.4f  (iv) unhedged loss "
      "%.2f\n",
      e27.position_value, e27.portfolio_value, e27.unhedged_loss);

  Header("Ex 28");
  const auto e28 = Ex28();
  std::printf("  (i) %.2f  (ii) Delta %.4f -> hold %.4f shares, cash %.2f\n",
              e28.value0, e28.port_delta0, e28.shares1, e28.cash1);
  std::printf("  (iii) value %.2f, hold %.4f shares (sell %.4f)\n", e28.value1,
              e28.shares2, e28.shares1 - e28.shares2);

  Header("Ex 29");
  const auto h29 = Ex29ComputeHedge();
  std::printf("  calls %.2f  puts %.2f  shares %.2f\n", h29.calls, h29.puts,
              h29.shares);
}
