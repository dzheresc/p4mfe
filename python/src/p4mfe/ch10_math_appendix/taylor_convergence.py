"""Section 10.8: convergence results for Taylor expansions.

If |f^(k)| <= C M^k between a and x for all k, then
    |R_n(x)| <= C (M |x - a|)^(n+1) / (n+1)!  -> 0,
so the Taylor series converges to f (e^x, sin, cos, e^(Mx)). Smoothness alone
is not enough: e^(-1/x^2) has all derivatives 0 at 0. A power series is the
Taylor series of its sum: a_k = f^(k)(c) / k!.
"""

from __future__ import annotations

import math
from typing import Sequence

from p4mfe.ch06_taylor_bonds_atm.taylor import taylor_remainder_bound
from p4mfe.ch10_math_appendix.power_series import PowerSeries


def taylor_bound_geometric(C: float, M: float, a: float, x: float, n: int) -> float:
    return taylor_remainder_bound(C * M ** (n + 1), a, x, n)


def taylor_terms_needed(C: float, M: float, a: float, x: float, tol: float, max_n: int = 1000) -> int:
    if not tol > 0:
        raise ValueError("tol must be positive")
    q = M * abs(x - a)
    bound = C * q
    for n in range(max_n + 1):
        if bound < tol:
            return n
        bound *= q / (n + 2)
    raise RuntimeError("tolerance not reached")


def flat_function(x: float) -> float:
    return 0.0 if x == 0 else math.exp(-1.0 / (x * x))


def taylor_series(derivs: Sequence[float], a: float) -> PowerSeries:
    return PowerSeries([d / math.factorial(k) for k, d in enumerate(derivs)], a)


def derivatives_at_center(s: PowerSeries) -> list[float]:
    return [c * math.factorial(k) for k, c in enumerate(s.a)]
