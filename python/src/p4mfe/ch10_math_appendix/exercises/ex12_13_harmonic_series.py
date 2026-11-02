"""Exercises 12-13.

12. sum 1/k^2 < 2 - 1/n converges; ln n + 1/n < H_n < ln n + 1 (n >= 2) diverges;
    I_n + 1/(n ln n) < sum_{k=2}^n 1/(k ln k) < I_n + 1/(2 ln 2),
    I_n = ln ln n - ln ln 2, diverges.
13. x_n = H_n - ln n decreases, y_n = H_n - ln(n+1) increases, x_n - y_n -> 0:
    the limit gamma lies in [1 - ln 2, 3/2 - ln 2] = [0.307, 0.807].

Full derivations: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex12_13_harmonic_series.hpp
"""

from __future__ import annotations

import math

EX13_EULER_GAMMA = 0.57721566490153286061


def ex12_sum_inv_squares(n: int) -> float:
    return math.fsum(1.0 / (k * k) for k in range(1, n + 1))


def ex12_harmonic(n: int) -> float:
    return math.fsum(1.0 / k for k in range(1, n + 1))


def ex12_sum_inv_k_log_k(n: int) -> float:
    return math.fsum(1.0 / (k * math.log(k)) for k in range(2, n + 1))


def ex12_harmonic_bounds(n: int) -> tuple[float, float]:
    l = math.log(n)
    return l + 1.0 / n, l + 1.0


def ex12_k_log_k_bounds(n: int) -> tuple[float, float]:
    integral = math.log(math.log(n)) - math.log(math.log(2.0))
    return integral + 1.0 / (n * math.log(n)), integral + 1.0 / (
        2.0 * math.log(2.0)
    )


def ex13_x(n: int) -> float:
    return ex12_harmonic(n) - math.log(n)


def ex13_y(n: int) -> float:
    return ex12_harmonic(n) - math.log(n + 1)
