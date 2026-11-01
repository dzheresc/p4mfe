#pragma once
// Chapter 10, section 10.4: sequences satisfying linear recursions.
//
// A sequence with x_{n+k} = a_{k-1} x_{n+k-1} + ... + a_0 x_n is fixed by its
// first k terms. Trying x_n = lambda^n gives the characteristic equation
//   lambda^k = a_{k-1} lambda^(k-1) + ... + a_0.
// For the second order recursion x_{n+2} = a x_{n+1} + b x_n with roots
// lambda_1, lambda_2 of lambda^2 - a lambda - b = 0:
//   distinct roots:  x_n = c_1 lambda_1^n + c_2 lambda_2^n
//   double root:     x_n = (c_1 + c_2 n) lambda^n
// with c_1, c_2 from x_0, x_1. Complex roots are allowed (the sequence stays
// real). Example: Fibonacci, x_n = (phi^n - (1-phi)^n) / sqrt(5).
//
// First order affine: x_{n+1} = a x_n + b has
//   x_n = a^n (x_0 - x*) + x*,  x* = b / (1 - a)   (a != 1),
// e.g. a loan balance B_{n+1} = (1 + r) B_n - P, which reaches zero after N
// payments when P = r B_0 / (1 - (1 + r)^(-N)).

#include <cmath>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch10 {

// x_0, ..., x_{count-1} for x_{n+k} = sum_{i<k} a[i] x_{n+i}, from init = x_0..x_{k-1}.
[[nodiscard]] inline std::vector<double> recursion_terms(std::span<const double> a, std::span<const double> init,
                                                         std::size_t count) {
    const std::size_t k = a.size();
    if (k == 0 || init.size() != k) throw std::invalid_argument("recursion_terms: need k coefficients and k initial terms");
    std::vector<double> x(init.begin(), init.end());
    x.resize(std::max(count, k));
    for (std::size_t n = k; n < count; ++n) {
        double v = 0.0;
        for (std::size_t i = 0; i < k; ++i) v += a[i] * x[n - k + i];
        x[n] = v;
    }
    x.resize(count);
    return x;
}

// Closed form of x_{n+2} = a x_{n+1} + b x_n.
struct SecondOrderRecursion {
    std::complex<double> lambda1, lambda2, c1, c2;
    bool double_root;

    // x_n (real part; the imaginary part is round-off).
    [[nodiscard]] double operator()(unsigned n) const {
        const auto p1 = power(lambda1, n);
        if (double_root) return ((c1 + c2 * static_cast<double>(n)) * p1).real();
        return (c1 * p1 + c2 * power(lambda2, n)).real();
    }

private:
    static std::complex<double> power(std::complex<double> z, unsigned n) {
        std::complex<double> r{1.0, 0.0};  // z^0 = 1, also for z = 0
        for (; n; n >>= 1, z *= z)
            if (n & 1u) r *= z;
        return r;
    }
};

// Roots of lambda^2 - a lambda - b and the constants matching x_0, x_1.
// `rel_tol` decides when the two roots count as one.
[[nodiscard]] inline SecondOrderRecursion solve_second_order(double a, double b, double x0, double x1,
                                                             double rel_tol = 1e-12) {
    const std::complex<double> disc = std::sqrt(std::complex<double>(a * a + 4.0 * b));
    const std::complex<double> l1 = 0.5 * (a + disc), l2 = 0.5 * (a - disc);
    SecondOrderRecursion r{l1, l2, {}, {}, std::abs(disc) <= rel_tol * std::max(1.0, std::abs(a))};
    if (r.double_root) {
        const double lambda = 0.5 * a;
        r.lambda1 = r.lambda2 = lambda;
        if (lambda == 0.0) throw std::invalid_argument("solve_second_order: a = b = 0 (x_n = 0 for n >= 2)");
        r.c1 = x0;
        r.c2 = x1 / lambda - x0;
    } else {
        r.c2 = (x1 - l1 * x0) / (l2 - l1);
        r.c1 = x0 - r.c2;
    }
    return r;
}

// x_n for x_{n+1} = a x_n + b.
[[nodiscard]] inline double first_order_affine(double a, double b, double x0, unsigned n) noexcept {
    if (a == 1.0) return x0 + n * b;
    const double fixed = b / (1.0 - a);
    return std::pow(a, n) * (x0 - fixed) + fixed;
}

// Level payment that amortizes a loan B_0 over N periods at rate r per period.
[[nodiscard]] inline double loan_payment(double B0, double r, unsigned N) {
    if (N == 0) throw std::invalid_argument("loan_payment: need N >= 1");
    if (r == 0.0) return B0 / N;
    return r * B0 / (1.0 - std::pow(1.0 + r, -static_cast<double>(N)));
}

}  // namespace p4mfe::ch10
