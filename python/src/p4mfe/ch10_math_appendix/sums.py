"""Section 10.3: useful sums with interesting proofs.

10.3.1  S_i(n) = sum_{k=1}^n k^i:  S_1 = n(n+1)/2, S_2 = n(n+1)(2n+1)/6,
        S_3 = S_1^2, and in general (telescoping (k+1)^(i+1) - k^(i+1))
            (n+1)^(i+1) - 1 = sum_{j=0}^{i} C(i+1, j) S_j(n).
        S_i is a polynomial of degree i+1 in n: the interpolating polynomial
        (10.2) through n = 0..i+1.
10.3.2  sum_{k=1}^n k^j x^k from the geometric sum by applying x d/dx;
        for |x| < 1, sum_{k>=1} k^j x^k = P_j(x) / (1-x)^(j+1) with P_0 = x,
        P_{j+1} = x (P_j' (1-x) + (j+1) P_j).

Python integers are exact, so the integer sums never overflow.
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch10_math_appendix.interpolation import interpolating_polynomial, polynomial_value


# 10.3.1 --------------------------------------------------------------------
def sum_of_powers(n: int, i: int) -> int:
    return sum(k**i for k in range(1, n + 1))


def sum_k(n: int) -> int:
    return n * (n + 1) // 2


def sum_k2(n: int) -> int:
    return n * (n + 1) * (2 * n + 1) // 6


def sum_k3(n: int) -> int:
    return sum_k(n) ** 2


def sums_of_powers_recursive(n: int, i: int) -> list[int]:
    """S_0(n), ..., S_i(n) from the telescoping recursion."""
    S = [n]
    for p in range(1, i + 1):
        rhs = (
            (n + 1) ** (p + 1)
            - 1
            - sum(math.comb(p + 1, j) * S[j] for j in range(p))
        )
        S.append(rhs // (p + 1))
    return S


def sum_of_powers_polynomial(i: int) -> np.ndarray:
    """Coefficients c_0..c_{i+1} of S_i(n) in n, by interpolation through m = 0..i+1."""
    ms = list(range(i + 2))
    return interpolating_polynomial(
        ms, [float(sum_of_powers(m, i)) for m in ms]
    )


# 10.3.2 --------------------------------------------------------------------
def sum_kj_xk(n: int, j: int, x: float) -> float:
    return sum(k**j * x**k for k in range(1, n + 1))


def geometric_sum(n: int, x: float) -> float:
    if x == 1.0:
        return float(n)
    return x * (1 - x**n) / (1 - x)


def sum_k_xk(n: int, x: float) -> float:
    if x == 1.0:
        return float(sum_k(n))
    xn = x**n
    return x * (1 - (n + 1) * xn + n * xn * x) / (1 - x) ** 2


def sum_k2_xk(n: int, x: float) -> float:
    if x == 1.0:
        return float(sum_k2(n))
    xn = x**n
    num = (
        1
        + x
        - (n + 1) ** 2 * xn
        + (2 * n * n + 2 * n - 1) * xn * x
        - n * n * xn * x * x
    )
    return x * num / (1 - x) ** 3


def kj_xk_numerator(j: int) -> list[float]:
    """P_j: P_0 = P_1 = x, P_2 = x + x^2, P_3 = x + 4x^2 + x^3 (Eulerian numbers)."""
    P = [0.0, 1.0]
    for m in range(j):
        Q = [0.0] * (len(P) + 1)
        for k, p in enumerate(P):
            Q[k + 1] += (m + 1) * p
            if k:
                Q[k] += k * p
                Q[k + 1] -= k * p
        while len(Q) > 1 and Q[-1] == 0.0:
            Q.pop()
        P = Q
    return P


def sum_kj_xk_infinite(j: int, x: float) -> float:
    if not abs(x) < 1:
        raise ValueError("need |x| < 1")
    return polynomial_value(kj_xk_numerator(j), x) / (1 - x) ** (j + 1)
