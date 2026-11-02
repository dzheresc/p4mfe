// Section 4.5: three routes to the Black-Scholes price of a call, all from
// e^(-rT) E_RN[(S_T - K)^+]: the closed form obtained by completing the
// square, numerical integration against the normal density, and Monte Carlo.
// Section 4.7: the call split into asset-or-nothing and cash-or-nothing
// digitals, and what N(d1) and N(d2) mean.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch04/in_the_money.hpp>
#include <p4mfe/ch04/risk_neutral.hpp>

int main() {
  using namespace p4mfe::ch04;
  const double S = 40, K = 42, T = 0.75, sigma = 0.3, r = 0.04, q = 0.01;
  std::printf("S = %g, K = %g, T = %g, sigma = %g, r = %g, q = %g\n\n", S, K, T,
              sigma, r, q);

  const auto call = [K](double s) { return std::max(s - K, 0.0); };
  std::printf("Black-Scholes (chapter 3)         %.10f\n",
              p4mfe::ch03::bs_call(S, K, T, sigma, r, q));
  std::printf("e^(-rT) x closed-form expectation %.10f\n",
              std::exp(-r * T) *
                  rn_expected_payoff(OptionType::Call, S, K, T, sigma, r, q));
  std::printf("e^(-rT) x integral                %.10f\n",
              risk_neutral_price(call, S, T, sigma, r, q));
  std::printf("\nMonte Carlo (antithetic):\n%10s %14s %12s\n", "paths", "price",
              "std error");
  for (std::size_t n = 1000; n <= 1'000'000; n *= 10) {
    const auto mc = monte_carlo_price(call, S, T, sigma, r, q, n);
    std::printf("%10zu %14.6f %12.6f\n", n, mc.price, mc.std_error);
  }

  const auto [n_d1, n_d2] = exercise_probabilities(S, K, T, sigma, r, q);
  const double aon = asset_or_nothing(OptionType::Call, S, K, T, sigma, r, q);
  const double con = cash_or_nothing(OptionType::Call, S, K, T, sigma, r, q);
  std::printf("\nasset-or-nothing call  S e^(-qT) N(d1) = %.6f\n", aon);
  std::printf("cash-or-nothing call   e^(-rT) N(d2)   = %.6f\n", con);
  std::printf("call = AoN - K CoN                     = %.6f\n", aon - K * con);
  std::printf("N(d2) = %.6f  risk-neutral probability the call is exercised\n",
              n_d2);
  std::printf(
      "N(d1) = %.6f  same, outcomes weighted by S_T; e^(-qT) N(d1) = Delta = "
      "%.6f\n",
      n_d1, std::exp(-q * T) * n_d1);
}
