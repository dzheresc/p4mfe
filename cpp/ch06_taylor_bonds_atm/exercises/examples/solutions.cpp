// Prints the numerical answers to the exercises of section 6.11.
//
//   ./build/cpp/ch06_taylor_bonds_atm/exercises/ch06_exercises_solutions

#include <p4mfe/ch06/exercises/ex01_sqrt_taylor.hpp>
#include <p4mfe/ch06/exercises/ex02_exp_six_digits.hpp>
#include <p4mfe/ch06/exercises/ex03_big_o.hpp>
#include <p4mfe/ch06/exercises/ex04_exp_composite_taylor.hpp>
#include <p4mfe/ch06/exercises/ex05_even_series.hpp>
#include <p4mfe/ch06/exercises/ex06_e_bounds.hpp>
#include <p4mfe/ch06/exercises/ex07_log_ratio_series.hpp>
#include <p4mfe/ch06/exercises/ex08_cosh_series.hpp>
#include <p4mfe/ch06/exercises/ex09_log_series_convergence.hpp>
#include <p4mfe/ch06/exercises/ex10_log_product_integral.hpp>
#include <p4mfe/ch06/exercises/ex11_crr_expansion.hpp>
#include <p4mfe/ch06/exercises/ex12_atm_put_error.hpp>
#include <p4mfe/ch06/exercises/ex13_atm_parity.hpp>
#include <p4mfe/ch06/exercises/ex14_atm_put_maturity.hpp>
#include <p4mfe/ch06/exercises/ex15_atm_put_dividends.hpp>
#include <p4mfe/ch06/exercises/ex16_bond_price_change.hpp>
#include <p4mfe/ch06/exercises/ex17_portfolio_immunization.hpp>
#include <p4mfe/ch06/exercises/ex18_curve_shift_yield.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>

using namespace p4mfe;
using namespace p4mfe::ch06;
using namespace p4mfe::ch06::ex;

namespace {
void header(const char* title) { std::printf("\n== %s ==\n", title); }
void print_put_row(double T, const Ex12Result& r) {
    std::printf("  T = %4.0f  P_approx = %10.6f  P_BS = %10.6f  rel. error = %8.4f%%\n", T, r.approx,
                r.black_scholes, 100 * r.relative_error);
}
}  // namespace

int main() {
    header("Ex 1: sqrt(1 + x) ~ 1 + x/2 - x^2/8 + x^3/16");
    for (double x : {0.1, 0.01})
        std::printf("  x = %.2f  sqrt = %.12f  P3 = %.12f  error/x^4 = %.6f (-5/128 = %.6f)\n", x,
                    std::sqrt(1 + x), ex01_cubic(x), (std::sqrt(1 + x) - ex01_cubic(x)) / std::pow(x, 4),
                    ex01_error_coefficient);

    header("Ex 2: e^0.25 to six decimals");
    const auto e2 = ex02_solve();
    std::printf("  n = %u  P_n(0.25) = %.10f  bound = %.3e  exact = %.10f\n", e2.n, e2.value, e2.bound,
                std::exp(0.25));

    header("Ex 3: e^(-x) - 1/(1 + x) = O(x^2)");
    for (double x : {0.1, 0.01, 0.001})
        std::printf("  x = %.3f  difference / x^2 = %.6f\n", x, ex03_difference(x) / (x * x));

    header("Ex 4: quadratic Taylor of e^((x+1)^2)");
    const auto c2 = ex04_part_ii(), c3 = ex04_part_iii();
    std::printf("  (ii)  %.6f + %.6f x + %.6f x^2\n", c2[0], c2[1], c2[2]);
    std::printf("  (iii) %.6f + %.6f x + %.6f x^2   (e, 2e, 3e)\n", c3[0], c3[1], c3[2]);

    header("Ex 5: ln(1 - x^2) and 1/(1 - x^2) at x = 0.5");
    std::printf("  series(40) = %.12f vs %.12f;  %.12f vs %.12f\n", ex05_log_series(0.5, 40), std::log1p(-0.25),
                ex05_inverse_series(0.5, 40), 1 / 0.75);

    header("Ex 6: (1+1/x)^(x+1/2-1/(12x)) < e < (1+1/x)^(x+1/2)");
    for (double x : {1.0, 2.0, 10.0, 100.0})
        std::printf("  x = %5.0f  %.12f < e = %.12f < %.12f\n", x, ex06_lower(x), std::numbers::e, ex06_upper(x));

    header("Ex 7: ln((1+x)/(1-x)) = 2 sum x^(2j+1)/(2j+1), R = 1");
    for (double x : {0.5, 0.9})
        std::printf("  x = %.1f  series(200) = %.12f  function = %.12f\n", x, ex07_series(x, 200), ex07_function(x));

    header("Ex 8: 1 + x^4/2! + x^8/4! + ... = cosh(x^2), R = infinity");
    for (double x : {1.0, 2.0})
        std::printf("  x = %.1f  series(30) = %.10f  cosh(x^2) = %.10f\n", x, ex08_series(x, 30), ex08_function(x));

    header("Ex 9: |ln(1+x) - P_n(x)| <= |x|^n |ln(1+x)|");
    for (double x : {-0.9, 0.9})
        for (unsigned n : {10u, 50u, 200u})
            std::printf("  x = %4.1f  n = %3u  error = %.3e  bound = %.3e\n", x, n, std::abs(ex09_error(x, n)),
                        ex09_bound(x, n));

    header("Ex 10: int_0^1 ln(1-x) ln(x) dx = 2 - pi^2/6");
    std::printf("  numerical = %.12f  series(1000) = %.12f  2 - pi^2/6 = %.12f\n", ex10_numerical(),
                ex10_partial_sum(1000), ex10_exact);

    header("Ex 11: CRR tree vs e^(+-sigma sqrt(dt)), r = 5%, sigma = 20%");
    for (double dt : {1e-2, 1e-3, 1e-4}) {
        const auto c = ex11_crr(0.05, 0.2, dt), a = ex11_approx(0.05, 0.2, dt);
        std::printf("  dt = %.0e  u - u~ = %.3e  d - d~ = %.3e  p - p~ = %.3e\n", dt, c.u - a.u, c.d - a.d, c.p - a.p);
    }

    header("Ex 12: ATM put, S = 60, sigma = 25%, T = 1");
    const auto z = ex12_zero_rates(), w = ex12_with_rates();
    std::printf("  r = q = 0:        approx %.6f  BS %.6f  rel. error %.6f\n", z.approx, z.black_scholes,
                z.relative_error);
    std::printf("  r = 6%%, q = 3%%:   approx %.6f  BS %.6f  rel. error %.6f\n", w.approx, w.black_scholes,
                w.relative_error);

    header("Ex 13: parity-consistent approximation (6.139)");
    const auto e13 = ex13_solve();
    std::printf("  approx %.6f  BS %.6f  rel. error %.6f (vs %.6f for 6.103)\n", e13.approx, e13.black_scholes,
                e13.relative_error, w.relative_error);

    header("Ex 14: ATM put, K = 40, sigma = 30%, r = q = 0");
    const auto t14 = ex14_table();
    for (std::size_t i = 0; i < t14.size(); ++i) print_put_row(ex14_maturities[i], t14[i]);

    header("Ex 15: ATM put, K = 40, sigma = 30%, r = 4.5%, q = 2%");
    const auto t15 = ex15_table();
    for (std::size_t i = 0; i < t15.size(); ++i) print_put_row(ex15_maturities[i], t15[i]);

    header("Ex 16: B = 101, D = 1.5, C = 2.5");
    for (const auto& row : ex16_table())
        std::printf("  dy = %.3f  (2.57) %.6f  (6.73) %.6f\n", row.dy, row.duration_only, row.duration_convexity);

    header("Ex 17: $50mil portfolio, DV01 $10,000, C$ $400mil");
    std::printf("  (i) new value after +50bp: %.2f\n", ex17_new_value());
    const auto h = ex17_immunize();
    std::printf("  (ii) bond 1 units %.4f  bond 2 units %.4f  cash %.2f\n", h.units1, h.units2, h.cash);

    header("Ex 18: 4-year 6% semiannual bond off r1(0, t)");
    const auto e18 = ex18_solve();
    std::printf("  B = %.6f  y = %.6f\n", e18.price, e18.yield);
    const auto dy = ex18_yield_changes();
    for (std::size_t i = 0; i < dy.size(); ++i)
        std::printf("  dr = %.3f  yield change = %.8f  (change - dr = %.4f bp)\n", ex18_shifts[i], dy[i],
                    (dy[i] - ex18_shifts[i]) * 1e4);
}
