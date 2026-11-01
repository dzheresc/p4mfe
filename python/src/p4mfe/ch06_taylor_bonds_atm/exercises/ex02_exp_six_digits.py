"""Exercise 2: e^0.25 to six decimal digits from the Taylor series of e^x.

Lagrange bound e^x - P_n(x) < 3^x x^(n+1)/(n+1)! (x > 0); n = 5 is the first with bound < 0.5e-6:
P_5(0.25) = 1.2840250651, so e^0.25 = 1.284025.

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex02_exp_six_digits.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..taylor_series import exp_series


@dataclass(frozen=True)
class Ex02Result:
    n: int
    value: float
    bound: float


def ex02_error_bound(x: float, n: int) -> float:
    return 3.0**x * x ** (n + 1) / math.factorial(n + 1)


def ex02_solve(x: float = 0.25, tol: float = 0.5e-6) -> Ex02Result:
    n = 0
    while ex02_error_bound(x, n) >= tol:
        n += 1
    return Ex02Result(n, exp_series(x, n), ex02_error_bound(x, n))
