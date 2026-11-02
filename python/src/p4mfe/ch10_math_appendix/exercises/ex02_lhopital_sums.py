"""Exercise 2: S(n, 2) and S(n, 3) as T(n, j, 1) by l'Hopital's rule.

T(n, j, x) = sum k^j x^k = N_j(x) / (1 - x)^(j+1), where x d/dx maps
N / (1-x)^p to x (N'(1-x) + p N) / (1-x)^(p+1). N_j vanishes to order j + 1 at
x = 1, so l'Hopital's rule applied j + 1 times gives
S(n, j) = N_j^(j+1)(1) / ((-1)^(j+1) (j+1)!); N_2'''(1) = -n(n+1)(2n+1),
N_3''''(1) = 6 n^2 (n+1)^2.

Full derivation: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex02_lhopital_sums.hpp
"""

from __future__ import annotations

import math

IntPoly = dict[int, int]  # exponent -> coefficient


def _clean(poly: IntPoly) -> IntPoly:
    return {m: c for m, c in sorted(poly.items()) if c != 0}


def ex02_x_ddx_numerator(numer: IntPoly, p: int) -> IntPoly:
    next_poly: IntPoly = {}
    for m, c in numer.items():
        next_poly[m] = next_poly.get(m, 0) + m * c
        next_poly[m + 1] = next_poly.get(m + 1, 0) + (p - m) * c
    return _clean(next_poly)


def ex02_numerator(n: int, j: int) -> IntPoly:
    if n < 1:
        raise ValueError("need n >= 1")
    numer: IntPoly = {1: 1}
    numer[n + 1] = numer.get(n + 1, 0) - 1
    for p in range(1, j + 1):
        numer = ex02_x_ddx_numerator(numer, p)
    return numer


def _add(poly: IntPoly, m: int, c: int) -> None:
    poly[m] = poly.get(m, 0) + c


def ex02_book_n2(n: int) -> IntPoly:
    poly: IntPoly = {}
    for m, c in (
        (1, 1),
        (2, 1),
        (n + 1, -((n + 1) ** 2)),
        (n + 2, 2 * n * n + 2 * n - 1),
        (n + 3, -n * n),
    ):
        _add(poly, m, c)
    return _clean(poly)


def ex02_book_n3(n: int) -> IntPoly:
    poly: IntPoly = {}
    terms = (
        (1, 1),
        (2, 4),
        (3, 1),
        (n + 1, -((n + 1) ** 3)),
        (n + 2, 3 * n**3 + 6 * n * n - 4),
        (n + 3, -(3 * n**3 + 3 * n * n - 3 * n + 1)),
        (n + 4, n**3),
    )
    for m, c in terms:
        _add(poly, m, c)
    return _clean(poly)


def ex02_derivative_at_one(poly: IntPoly, r: int) -> int:
    """d^r/dx^r P at x = 1 (exact)."""
    return sum(c * math.perm(m, r) if m >= r else 0 for m, c in poly.items())


def ex02_sum_by_lhopital(n: int, j: int) -> int:
    denom = (-1) ** (j + 1) * math.factorial(j + 1)
    return ex02_derivative_at_one(ex02_numerator(n, j), j + 1) // denom


def ex02_t_closed_form(n: int, j: int, x: float) -> float:
    num = sum(c * x**m for m, c in ex02_numerator(n, j).items())
    return num / (1 - x) ** (j + 1)
