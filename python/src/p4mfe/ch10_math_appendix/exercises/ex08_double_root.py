"""Exercise 8: sum a_i y_{n+i} = C n lambda^n P(lambda) + C lambda^(n+1) P'(lambda)
for y_n = C n lambda^n, so a double root (P = P' = 0) gives a solution.

Full derivation: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex08_double_root.hpp
"""

from __future__ import annotations

from typing import Sequence

from p4mfe.ch10_math_appendix.interpolation import polynomial_value


def ex08_residual(a: Sequence[float], lam: float, c: float, n: int) -> float:
    return sum(ai * c * (n + i) * lam ** (n + i) for i, ai in enumerate(a))


def ex08_hint(a: Sequence[float], lam: float, c: float, n: int) -> float:
    d_poly = sum(i * a[i] * lam ** (i - 1) for i in range(1, len(a)))
    return (
        c * n * lam**n * polynomial_value(a, lam) + c * lam ** (n + 1) * d_poly
    )
