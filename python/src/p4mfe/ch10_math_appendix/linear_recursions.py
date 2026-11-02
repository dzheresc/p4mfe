"""Section 10.4: sequences satisfying linear recursions.

x_{n+k} = a_{k-1} x_{n+k-1} + ... + a_0 x_n. For x_{n+2} = a x_{n+1} + b x_n
with roots of lambda^2 - a lambda - b = 0:
    distinct roots  x_n = c1 l1^n + c2 l2^n   (complex roots allowed)
    double root     x_n = (c1 + c2 n) l^n.
First order affine x_{n+1} = a x_n + b: x_n = a^n (x_0 - x*) + x*,
x* = b / (1 - a); a loan B_{n+1} = (1+r) B_n - P is repaid in N periods when
P = r B_0 / (1 - (1+r)^(-N)).
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from typing import Sequence


def recursion_terms(
    a: Sequence[float], init: Sequence[float], count: int
) -> list[float]:
    """x_0..x_{count-1} for x_{n+k} = sum_{i<k} a[i] x_{n+i}."""
    k = len(a)
    if k == 0 or len(init) != k:
        raise ValueError("need k coefficients and k initial terms")
    x = [float(v) for v in init]
    while len(x) < count:
        x.append(sum(a[i] * x[-k + i] for i in range(k)))
    return x[:count]


@dataclass(frozen=True)
class SecondOrderRecursion:
    lambda1: complex
    lambda2: complex
    c1: complex
    c2: complex
    double_root: bool

    def __call__(self, n: int) -> float:
        if self.double_root:
            return ((self.c1 + self.c2 * n) * self.lambda1**n).real
        return (self.c1 * self.lambda1**n + self.c2 * self.lambda2**n).real


def solve_second_order(
    a: float, b: float, x0: float, x1: float, rel_tol: float = 1e-12
) -> SecondOrderRecursion:
    disc = cmath.sqrt(a * a + 4 * b)
    l1, l2 = (a + disc) / 2, (a - disc) / 2
    if abs(disc) <= rel_tol * max(1.0, abs(a)):
        lam = a / 2
        if lam == 0:
            raise ValueError("a = b = 0 (x_n = 0 for n >= 2)")
        return SecondOrderRecursion(
            complex(lam),
            complex(lam),
            complex(x0),
            complex(x1 / lam - x0),
            True,
        )
    c2 = (x1 - l1 * x0) / (l2 - l1)
    return SecondOrderRecursion(l1, l2, x0 - c2, c2, False)


def first_order_affine(a: float, b: float, x0: float, n: int) -> float:
    if a == 1.0:
        return x0 + n * b
    fixed = b / (1 - a)
    return a**n * (x0 - fixed) + fixed


def loan_payment(principal: float, r: float, num_periods: int) -> float:
    if num_periods < 1:
        raise ValueError("need N >= 1")
    if r == 0:
        return principal / num_periods
    return r * principal / (1 - (1 + r) ** (-num_periods))
