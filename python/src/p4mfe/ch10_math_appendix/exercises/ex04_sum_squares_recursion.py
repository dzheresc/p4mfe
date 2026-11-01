"""Exercise 4: x_n = sum_{k<=n} k^2 satisfies
x_{n+4} - 4x_{n+3} + 6x_{n+2} - 4x_{n+1} + x_n = 0, with characteristic
polynomial (z - 1)^4; so x_n is the cubic through (1, 1), (2, 5), (3, 14),
(4, 30), which is n(n+1)(2n+1)/6.

Full derivation: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex04_sum_squares_recursion.hpp
"""

from __future__ import annotations

import numpy as np

from p4mfe.ch10_math_appendix.interpolation import interpolating_polynomial
from p4mfe.ch10_math_appendix.linear_recursions import recursion_terms

EX04_RECURSION = (-1.0, 4.0, -6.0, 4.0)  # x_{n+4} = sum a_i x_{n+i}


def ex04_terms(count: int) -> list[float]:
    """x_1..x_count (index 0 holds x_1)."""
    return recursion_terms(EX04_RECURSION, [1.0, 5.0, 14.0, 30.0], count)


def ex04_terms_10_93(count: int) -> list[float]:
    x = [1.0, 5.0]
    n = 1
    while len(x) < count:
        x.append(2 * x[n] - x[n - 1] + 2 * n + 3)
        n += 1
    return x[:count]


def ex04_cubic() -> np.ndarray:
    return interpolating_polynomial([1.0, 2.0, 3.0, 4.0], [1.0, 5.0, 14.0, 30.0])
