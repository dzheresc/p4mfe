"""Exercise 3: S(n, 4) = n(n+1)(2n+1)(3n^2+3n-1)/30 = n^5/5 + n^4/2 + n^3/3 - n/30.

From (n+1)^5 - 1 = S(n,0) + 5 S(n,1) + 10 S(n,2) + 10 S(n,3) + 5 S(n,4).

Full derivation: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex03_sum_k4.hpp
"""

from __future__ import annotations

from p4mfe.ch10_math_appendix.sums import sums_of_powers_recursive


def ex03_sum_k4(n: int) -> int:
    return n * (n + 1) * (2 * n + 1) * (3 * n * n + 3 * n - 1) // 30


def ex03_sum_k4_recursive(n: int) -> int:
    return sums_of_powers_recursive(n, 4)[4]
