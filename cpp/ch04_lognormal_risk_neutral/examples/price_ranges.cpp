// Sections 4.4 and 4.6: in the lognormal model, the range S_T falls in with a
// given probability, and the probability that options expire in the money
// under the real-world drift mu versus the risk-neutral drift r.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch04/asset_model.hpp>
#include <p4mfe/ch04/in_the_money.hpp>

int main() {
  using namespace p4mfe::ch04;
  const double S0 = 50, mu = 0.10, sigma = 0.25, q = 0.0, r = 0.04;
  std::printf("S0 = %g, mu = %g, sigma = %g, q = %g\n\n", S0, mu, sigma, q);

  std::printf("%6s %10s %10s %10s %18s %18s\n", "T", "median", "mean", "stddev",
              "95% range", "99% range");
  for (double T : {1.0 / 12, 0.5, 1.0, 2.0}) {
    const auto r95 = price_confidence_interval(0.95, S0, T, sigma, mu, q);
    const auto r99 = price_confidence_interval(0.99, S0, T, sigma, mu, q);
    std::printf(
        "%6.3f %10.4f %10.4f %10.4f   [%6.2f, %6.2f]   [%6.2f, %6.2f]\n", T,
        price_median(S0, T, sigma, mu, q), price_mean(S0, T, mu, q),
        std::sqrt(price_variance(S0, T, sigma, mu, q)), r95.lo, r95.hi, r99.lo,
        r99.hi);
  }

  const double T = 0.5;
  std::printf("\nT = %g: probability of finishing in the money\n", T);
  std::printf("%6s %14s %14s %14s %14s\n", "K", "call (mu)", "call (r)",
              "put (mu)", "put (r)");
  for (double K : {40.0, 45.0, 50.0, 55.0, 60.0}) {
    std::printf("%6g %14.6f %14.6f %14.6f %14.6f\n", K,
                prob_itm(OptionType::kCall, S0, K, T, sigma, mu, q),
                prob_itm(OptionType::kCall, S0, K, T, sigma, r, q),
                prob_itm(OptionType::kPut, S0, K, T, sigma, mu, q),
                prob_itm(OptionType::kPut, S0, K, T, sigma, r, q));
  }
}
