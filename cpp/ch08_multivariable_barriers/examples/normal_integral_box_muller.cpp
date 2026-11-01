// Sections 8.3.1, 8.6 and 8.7: the integral of e^(-x^2/2) through polar
// coordinates, and Box-Muller normal samples.

#include <p4mfe/ch08/box_muller.hpp>
#include <p4mfe/ch08/double_integrals.hpp>
#include <p4mfe/normal.hpp>

#include <cmath>
#include <cstdio>
#include <numbers>

int main() {
    using namespace p4mfe;
    using namespace p4mfe::ch08;

    std::printf("Integral of e^(-(x^2+y^2)/2): disk(R) <= square(R) <= disk(R sqrt 2) -> 2 pi = %.12f\n",
                2 * std::numbers::pi);
    std::printf("%6s  %16s  %16s  %16s  %16s\n", "R", "disk (polar num)", "disk (closed)", "square", "disk(R sqrt 2)");
    for (double R : {0.5, 1.0, 2.0, 4.0, 8.0})
        std::printf("%6.1f  %16.12f  %16.12f  %16.12f  %16.12f\n", R,
                    polar_integral([](double x, double y) { return std::exp(-0.5 * (x * x + y * y)); }, 0, R),
                    gaussian_disk_integral(R), gaussian_square_integral(R),
                    gaussian_disk_integral(R * std::numbers::sqrt2));

    std::printf("\nBox-Muller samples: moments and tail probabilities\n");
    std::printf("%10s  %10s  %10s  %10s  %12s\n", "n", "mean", "variance", "kurtosis", "P(Z < -2)");
    for (std::size_t n : {1'000, 10'000, 100'000, 1'000'000}) {
        const auto z = box_muller_samples(n, 2024);
        double m = 0, m2 = 0, m4 = 0, tail = 0;
        for (double x : z) {
            m += x;
            m2 += x * x;
            m4 += x * x * x * x;
            tail += x < -2.0;
        }
        const double N = static_cast<double>(n);
        std::printf("%10zu  %10.5f  %10.5f  %10.5f  %12.6f\n", n, m / N, m2 / N, m4 / N, tail / N);
    }
    std::printf("%10s  %10.5f  %10.5f  %10.5f  %12.6f\n", "exact", 0.0, 1.0, 3.0, norm_cdf(-2.0));
}
