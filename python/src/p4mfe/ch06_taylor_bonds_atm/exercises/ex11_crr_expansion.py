"""Exercise 11: for small dt the CRR tree has u ~ e^(sigma sqrt(dt)), d ~ e^(-sigma sqrt(dt)),
p ~ 1/2 + (1/2)(r/sigma - sigma/2) sqrt(dt); the expansions agree up to O(dt^(3/2)).

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex11_crr_expansion.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Ex11Tree:
    u: float
    d: float
    p: float


def ex11_a(r: float, sigma: float, dt: float) -> float:
    return 0.5 * (math.exp(-r * dt) + math.exp((r + sigma**2) * dt))


def ex11_crr(r: float, sigma: float, dt: float) -> Ex11Tree:
    a = ex11_a(r, sigma, dt)
    am1 = 0.5 * (math.expm1(-r * dt) + math.expm1((r + sigma**2) * dt))
    root = math.sqrt(am1 * (a + 1.0))
    u, d = a + root, a - root
    return Ex11Tree(u, d, (math.exp(r * dt) - d) / (u - d))


def ex11_approx(r: float, sigma: float, dt: float) -> Ex11Tree:
    s = sigma * math.sqrt(dt)
    return Ex11Tree(
        math.exp(s),
        math.exp(-s),
        0.5 + 0.5 * (r / sigma - 0.5 * sigma) * math.sqrt(dt),
    )


def ex11_u_coefficient(r: float, sigma: float) -> float:
    """Leading coefficient of u_CRR - e^(sigma sqrt(dt)) in powers of dt^(3/2)."""
    return r * (r + sigma**2) / (2.0 * sigma) + 5.0 * sigma**3 / 24.0
