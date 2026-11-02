"""Exercise 11: g(x) = 1/sqrt(2 pi) int_0^b(x) e^(-y^2/2) dy,
b(x) = (ln(x/K) + (r + sigma^2/2) T) / (sigma sqrt T).

g'(x) = e^(-b(x)^2/2) b'(x) / sqrt(2 pi) = e^(-b(x)^2/2) / (x sigma sqrt(2 pi T)),
the Gamma of a call; g(x) = N(b(x)) - 1/2 = Delta_call - 1/2.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from ...normal import norm_cdf


class Ex11Params(NamedTuple):
    K: float
    T: float
    sigma: float
    r: float


def ex11_b(p: Ex11Params, x: float) -> float:
    return (math.log(x / p.K) + (p.r + 0.5 * p.sigma**2) * p.T) / (
        p.sigma * math.sqrt(p.T)
    )


def ex11_g(p: Ex11Params, x: float) -> float:
    return norm_cdf(ex11_b(p, x)) - 0.5


def ex11_g_prime(p: Ex11Params, x: float) -> float:
    b = ex11_b(p, x)
    return math.exp(-0.5 * b * b) / (
        x * p.sigma * math.sqrt(2.0 * math.pi * p.T)
    )
