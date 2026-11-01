"""Exercise 11: sum C(n, j) = 2^n, sum C(n, 2k) = 2^(n-1), sum j C(n, j) = n 2^(n-1)
(a = b = 1; a = -1, b = 1 added to the first; d/da of (a + 1)^n at a = 1).

Full derivation: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex11_binomial_identities.hpp
"""

from __future__ import annotations

from math import comb


def ex11_sum_binomials(n: int) -> int:
    return sum(comb(n, j) for j in range(n + 1))


def ex11_sum_even_binomials(n: int) -> int:
    return sum(comb(n, 2 * k) for k in range(n // 2 + 1))


def ex11_sum_j_binomials(n: int) -> int:
    return sum(j * comb(n, j) for j in range(n + 1))
