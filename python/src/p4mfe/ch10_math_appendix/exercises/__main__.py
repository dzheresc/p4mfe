"""Print the answers to the chapter 10 exercises.

python -m p4mfe.ch10_math_appendix.exercises
"""

from __future__ import annotations

import math

from p4mfe.ch10_math_appendix.parity import parity

from .ex01_parity import ex01_h
from .ex02_lhopital_sums import ex02_derivative_at_one, ex02_numerator, ex02_sum_by_lhopital
from .ex03_sum_k4 import ex03_sum_k4, ex03_sum_k4_recursive
from .ex04_sum_squares_recursion import ex04_cubic, ex04_terms
from .ex05_07_linear_recursions import ex05_constants, ex06_closed_form, ex07_closed_form, ex07_constants
from .ex08_double_root import ex08_residual
from .ex09_10_big_o import (
    ex09_f_big,
    ex09_f_small,
    ex09_g_big,
    ex09_g_small,
    ex10_ratio_k2,
    ex10_ratio_k3,
    ex10_remainder_ratio_k2,
    ex10_remainder_ratio_k3,
)
from .ex11_binomial_identities import ex11_sum_binomials, ex11_sum_even_binomials, ex11_sum_j_binomials
from .ex12_13_harmonic_series import (
    EX13_EULER_GAMMA,
    ex12_harmonic,
    ex12_harmonic_bounds,
    ex12_k_log_k_bounds,
    ex12_sum_inv_k_log_k,
    ex12_sum_inv_squares,
    ex13_x,
)
from .ex14_power_series_radius import ex14_partial_sum, ex14_ratio
from .ex15_17_nested_limits import (
    EX17_X_MAX,
    ex15_iterate,
    ex15_limit,
    ex16_iterate,
    ex16_limit,
    ex17_tower,
    ex17_x_for,
)


def header(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> None:
    header("Ex 1: parity of x^i f(x^j), f = sin")
    xs = [0.2, 0.5, 1.0, 1.5, 2.0]
    for i in range(1, 4):
        row = [
            f"i={i} j={j}: {parity(lambda x: ex01_h(math.sin, i, j, x), xs, 1e-10).value:<5}"
            for j in range(1, 4)
        ]
        print("  " + "  ".join(row))

    header("Ex 2: S(n, 2) and S(n, 3) by l'Hopital's rule at x = 1")
    for n in (1, 5, 10, 100):
        d3 = ex02_derivative_at_one(ex02_numerator(n, 2), 3)
        print(
            f"  n = {n:<4} N2'''(1) = {d3:<10} S(n,2) = {ex02_sum_by_lhopital(n, 2):<8} S(n,3) = {ex02_sum_by_lhopital(n, 3)}"
        )

    header("Ex 3: S(n, 4) = n(n+1)(2n+1)(3n^2+3n-1)/30")
    for n in (1, 2, 3, 10, 100):
        print(
            f"  n = {n:<4} closed form {ex03_sum_k4(n):<12} recursion {ex03_sum_k4_recursive(n)}"
        )

    header("Ex 4: x_{n+4} - 4x_{n+3} + 6x_{n+2} - 4x_{n+1} + x_n = 0")
    print("  x_1..x_8:", " ".join(f"{v:.0f}" for v in ex04_terms(8)))
    c4 = ex04_cubic()
    print(
        "  cubic coefficients c_0..c_3:",
        " ".join(f"{v:.6f}" for v in c4),
        " (0, 1/6, 1/2, 1/3)",
    )

    header("Ex 5: x_{n+3} = 2x_{n+1} + x_n, x_0 = 1, x_1 = -1, x_2 = 1")
    c5 = ex05_constants()
    print(
        f"  (c1, c2, c3) = ({c5[0]:.3g}, {abs(c5[1]):.3g}, {abs(c5[2]):.3g}): x_n = (-1)^n"
    )

    header("Ex 6: x_{n+1} = 3x_n + 2, x_0 = 1")
    print(
        "  x_n = 2 * 3^n - 1:",
        " ".join(f"{ex06_closed_form(n):.0f}" for n in range(7)),
    )

    header("Ex 7: x_{n+1} = 3x_n + n + 2, x_0 = 1")
    c7 = ex07_constants()
    print(
        f"  (c1, c2, c3) = ({c7[0]:.4f}, {c7[1]:.4f}, {c7[2]:.4f}) = (-5/4, -1/2, 9/4)"
    )
    print(
        "  x_n = (3^(n+2) - 2n - 5)/4:",
        " ".join(f"{ex07_closed_form(n):.0f}" for n in range(7)),
    )

    header("Ex 8: P(z) = (z-2)^2 (z+1), y_n = n 2^n")
    a8 = [4.0, 0.0, -3.0, 1.0]
    for n in (0, 5, 10):
        print(
            f"  n = {n:<3} residual at 2: {ex08_residual(a8, 2.0, 1.0, n):.3g}, at the simple root -1: {ex08_residual(a8, -1.0, 1.0, n):.3g}"
        )

    header(
        "Ex 9: |(f+g)/x^2| for f = x^2 sin(1/x), g = 3x^2, and for f = x^3, g = x^2.5"
    )
    for x in (1e-1, 1e-2, 1e-3, 1e-4):
        big = abs(ex09_f_big(x) + ex09_g_big(x)) / x**2
        small = (ex09_f_small(x) + ex09_g_small(x)) / x**2
        print(f"  x = {x:<6g} O-sum ratio {big:.4f}   o-sum ratio {small:.4g}")

    header("Ex 10: ratios that stay bounded")
    for n in (1, 10, 100, 1000, 10000):
        print(
            f"  n = {n:<6} {ex10_ratio_k2(n):.6f} {ex10_remainder_ratio_k2(n):.6f} "
            f"{ex10_ratio_k3(n):.6f} {ex10_remainder_ratio_k3(n):.6f}"
        )

    header("Ex 11: binomial identities")
    for n in (1, 5, 10, 20):
        print(
            f"  n = {n:<3} {ex11_sum_binomials(n)} {ex11_sum_even_binomials(n)} {ex11_sum_j_binomials(n)}"
        )

    header("Ex 12: partial sums")
    for n in (10, 1000, 100000):
        hl, hh = ex12_harmonic_bounds(n)
        kl, kh = ex12_k_log_k_bounds(n)
        print(
            f"  n = {n:<7} sum 1/k^2 = {ex12_sum_inv_squares(n):.8f} (pi^2/6 = {math.pi**2 / 6:.8f})  "
            f"{hl:.4f} < H_n = {ex12_harmonic(n):.4f} < {hh:.4f}  "
            f"{kl:.4f} < sum 1/(k ln k) = {ex12_sum_inv_k_log_k(n):.4f} < {kh:.4f}"
        )

    header("Ex 13: x_n = H_n - ln n")
    for n in (1, 2, 10, 1000, 100000):
        print(
            f"  n = {n:<7} x_n = {ex13_x(n):.10f}  x_n - gamma = {ex13_x(n) - EX13_EULER_GAMMA:.3e}  1/(2n) = {0.5 / n:.3e}"
        )

    header("Ex 14: sum x^k / (k ln k), R = 1")
    for k in (10, 1000, 1000000):
        print(f"  |a_k/a_(k+1)| at k = {k:<8} {ex14_ratio(k):.8f}")
    print(
        f"  x = 1:  S_n = {ex14_partial_sum(1.0, 100):.4f}, {ex14_partial_sum(1.0, 10000):.4f} for n = 10^2, 10^4"
    )
    print(
        f"  x = -1: S_n = {ex14_partial_sum(-1.0, 100000):.8f}, {ex14_partial_sum(-1.0, 100001):.8f} for n = 10^5, 10^5 + 1"
    )

    header("Ex 15: sqrt(a + sqrt(a + ...)) = (1 + sqrt(1 + 4a))/2")
    for a in (1.0, 2.0, 6.0, 10.0):
        print(
            f"  a = {a:<4g} limit {ex15_limit(a):.12f}  50 iterations {ex15_iterate(a, 50):.12f}"
        )

    header("Ex 16: a + 1/(a + 1/(a + ...)) = (a + sqrt(a^2 + 4))/2")
    for a in (0.5, 1.0, 2.0, 10.0):
        print(
            f"  a = {a:<4g} limit {ex16_limit(a):.12f}  50 iterations {ex16_iterate(a, 50):.12f}"
        )

    header("Ex 17: power towers")
    print(
        f"  (i)  x = sqrt 2 = {ex17_x_for(2.0):.12f}, tower after 100 levels {ex17_tower(ex17_x_for(2.0), 100):.12f}"
    )
    print(
        f"  (ii) largest x = e^(1/e) = {EX17_X_MAX:.12f}, largest b = e = {math.e:.12f}"
    )
    for n in (100, 10000, 100000):
        t = ex17_tower(EX17_X_MAX, n)
        print(
            f"       tower of e^(1/e) after {n:<6} levels {t:.10f}  (e - t_n) n / (2e) = {(math.e - t) * n / (2 * math.e):.4f}"
        )


if __name__ == "__main__":
    main()
