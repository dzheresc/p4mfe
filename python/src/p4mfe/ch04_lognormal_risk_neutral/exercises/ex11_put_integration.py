"""Exercise 11: a European put as a finite risk-neutral integral (Midpoint and Simpson vs Black-Scholes).

P = e^(-rT) Integral_0^K (K - x) f(x) dx, f the risk-neutral lognormal density of S_T.
K = 50, T = 0.5, q = 2%, r = 4%. Assumption: the text omits spot and volatility; we take S0 = 50, sigma = 0.3.
P_BS = 3.918999; starting at n = 4 and doubling until successive values differ by < 5e-7, the
Midpoint rule stops at n = 8192 and Simpson's rule at n = 128.

Full derivations: cpp/ch04_lognormal_risk_neutral/exercises/include/p4mfe/ch04/exercises/ex11_put_integration.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch02_integration_bonds.integration import Rule, integrate
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_put
from p4mfe.ch04_lognormal_risk_neutral.asset_model import price_pdf


@dataclass(frozen=True)
class Ex11Market:
    spot0: float = 50.0
    strike: float = 50.0
    maturity: float = 0.5
    sigma: float = 0.3
    r: float = 0.04
    q: float = 0.02


def ex11_integrand(x, m: Ex11Market = Ex11Market()):
    """e^(-rT) (K - x) f(x) on [0, K]; accepts arrays."""
    return (
        math.exp(-m.r * m.maturity)
        * (m.strike - x)
        * price_pdf(x, m.spot0, m.maturity, m.sigma, m.r, m.q)
    )


def ex11_put_bs(m: Ex11Market = Ex11Market()) -> float:
    return bs_put(m.spot0, m.strike, m.maturity, m.sigma, m.r, m.q)


def ex11_put_numeric(n: int, rule: Rule, m: Ex11Market = Ex11Market()) -> float:
    return integrate(lambda x: ex11_integrand(x, m), 0.0, m.strike, n, rule)


@dataclass(frozen=True)
class Ex11Step:
    n: int
    value: float
    error: float  # value - P_BS


def ex11_run(
    rule: Rule, tol: float = 5e-7, n0: int = 4, m: Ex11Market = Ex11Market()
) -> list[Ex11Step]:
    exact = ex11_put_bs(m)
    steps: list[Ex11Step] = []
    n = n0
    while n <= 1 << 24:
        v = ex11_put_numeric(n, rule, m)
        steps.append(Ex11Step(n, v, v - exact))
        if len(steps) > 1 and abs(v - steps[-2].value) < tol:
            break
        n *= 2
    return steps
