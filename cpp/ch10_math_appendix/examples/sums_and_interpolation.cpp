// Sections 10.2-10.3: sum_{k=1}^n k^i as the interpolating polynomial through
// n = 0..i+1, and the sums sum k^j x^k behind duration and convexity.

#include <p4mfe/ch10/interpolation.hpp>
#include <p4mfe/ch10/sums.hpp>

#include <cmath>
#include <cstdio>

int main() {
    using namespace p4mfe::ch10;

    std::printf("S_i(n) = sum_{k=1}^n k^i as a polynomial in n (interpolated through n = 0..i+1)\n");
    for (unsigned i = 1; i <= 5; ++i) {
        const auto c = sum_of_powers_polynomial(i);
        std::printf("  i = %u:", i);
        for (std::size_t m = c.size(); m-- > 1;) std::printf("  %+.6f n^%zu", c[m], m);
        std::printf("   (constant %.1e)\n", c[0]);
    }

    std::printf("\nTelescoping recursion vs. direct sums, n = 100\n");
    const auto S = sums_of_powers_recursive(100, 5);
    for (unsigned i = 0; i <= 5; ++i)
        std::printf("  i = %u  %16llu  %16llu\n", i, static_cast<unsigned long long>(S[i]),
                    static_cast<unsigned long long>(sum_of_powers(100, i)));

    std::printf("\nsum_{k>=1} k^j x^k = P_j(x) / (1-x)^(j+1)\n");
    for (unsigned j = 0; j <= 5; ++j) {
        const auto P = kj_xk_numerator(j);
        std::printf("  j = %u  P_j:", j);
        for (std::size_t k = 1; k < P.size(); ++k) std::printf(" %g", P[k]);
        std::printf("   at x = 0.9: %.6f\n", sum_kj_xk_infinite(j, 0.9));
    }

    std::printf("\nAnnual coupon bond, 10 years, 5%% coupon, yield 6%% (continuous): x = e^(-y)\n");
    const double x = std::exp(-0.06), c = 5.0, F = 100.0;
    const unsigned n = 10;
    const double B = c * geometric_sum(n, x) + F * std::pow(x, n);
    const double D = (c * sum_k_xk(n, x) + n * F * std::pow(x, n)) / B;
    const double C = (c * sum_k2_xk(n, x) + n * n * F * std::pow(x, n)) / B;
    std::printf("  price %.6f  duration %.6f  convexity %.6f\n", B, D, C);
}
