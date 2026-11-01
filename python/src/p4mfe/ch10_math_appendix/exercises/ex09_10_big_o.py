"""Exercises 9-10.

9.  O(x^n) + O(x^n) = O(x^n) and o(x^n) + o(x^n) = o(x^n) as x -> 0
    (triangle inequality; limit of a sum).
10. sum k^2 / n^3 = 1/3 + 1/(2n) + 1/(6n^2) <= 1, (sum k^2 - n^3/3)/n^2 = 1/2 + 1/(6n);
    sum k^3 / n^4 = 1/4 + 1/(2n) + 1/(4n^2) <= 1, (sum k^3 - n^4/4)/n^3 = 1/2 + 1/(4n).

Full derivations: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex09_10_big_o.hpp
"""

from __future__ import annotations

import math

from p4mfe.ch10_math_appendix.sums import sum_k2, sum_k3


def ex09_f_big(x: float) -> float:
    return x * x * math.sin(1 / x)


def ex09_g_big(x: float) -> float:
    return 3 * x * x


def ex09_f_small(x: float) -> float:
    return x**3


def ex09_g_small(x: float) -> float:
    return x * x * math.sqrt(x)


def ex10_ratio_k2(n: int) -> float:
    return sum_k2(n) / n**3


def ex10_remainder_ratio_k2(n: int) -> float:
    return (3 * sum_k2(n) - n**3) / (3 * n * n)


def ex10_ratio_k3(n: int) -> float:
    return sum_k3(n) / n**4


def ex10_remainder_ratio_k3(n: int) -> float:
    return (4 * sum_k3(n) - n**4) / (4 * n**3)
