// Sections 10.2-10.3: sum_{k=1}^n k^i as the interpolating polynomial through
// n = 0..i+1, and the sums sum k^j x^k behind duration and convexity.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch10/interpolation.hpp>
#include <p4mfe/ch10/sums.hpp>

int main() {
  using namespace p4mfe::ch10;

  std::printf(
      "S_i(n) = sum_{k=1}^n k^i as a polynomial in n (interpolated through n = "
      "0..i+1)\n");
  for (unsigned i = 1; i <= 5; ++i) {
    const auto c = SumOfPowersPolynomial(i);
    std::printf("  i = %u:", i);
    for (std::size_t m = c.size(); m-- > 1;)
      std::printf("  %+.6f n^%zu", c[m], m);
    std::printf("   (constant %.1e)\n", c[0]);
  }

  std::printf("\nTelescoping recursion vs. direct sums, n = 100\n");
  const auto sums = SumsOfPowersRecursive(100, 5);
  for (unsigned i = 0; i <= 5; ++i)
    std::printf("  i = %u  %16llu  %16llu\n", i,
                static_cast<unsigned long long>(sums[i]),
                static_cast<unsigned long long>(SumOfPowers(100, i)));

  std::printf("\nsum_{k>=1} k^j x^k = P_j(x) / (1-x)^(j+1)\n");
  for (unsigned j = 0; j <= 5; ++j) {
    const auto numer = KjXkNumerator(j);
    std::printf("  j = %u  P_j:", j);
    for (std::size_t k = 1; k < numer.size(); ++k) std::printf(" %g", numer[k]);
    std::printf("   at x = 0.9: %.6f\n", SumKjXkInfinite(j, 0.9));
  }

  std::printf(
      "\nAnnual coupon bond, 10 years, 5%% coupon, yield 6%% (continuous): x = "
      "e^(-y)\n");
  const double x = std::exp(-0.06), c = 5.0, face = 100.0;
  const unsigned n = 10;
  const double price = c * GeometricSum(n, x) + face * std::pow(x, n);
  const double duration =
      (c * SumKXk(n, x) + n * face * std::pow(x, n)) / price;
  const double convexity =
      (c * SumK2Xk(n, x) + n * n * face * std::pow(x, n)) / price;
  std::printf("  price %.6f  duration %.6f  convexity %.6f\n", price, duration,
              convexity);
}
