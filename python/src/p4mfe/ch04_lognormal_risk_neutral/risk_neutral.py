"""Section 4.5: the risk-neutral derivation of the Black-Scholes formula.

Under the risk-neutral measure mu = r and the value of a payoff g(S_T) is
    V = e^(-rT) E_RN[g(S_T)] = e^(-rT) int g(S_T(z)) phi(z) dz.
For the call, completing the square on z > -d2 gives
    E_RN[(S_T - K)^+] = S_0 e^((r - q) T) N(d1) - K N(d2).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_d1_d2
from p4mfe.ch03_probability_black_scholes.probability import ContinuousDistribution
from p4mfe.ch04_lognormal_risk_neutral.asset_model import price_from_normal
from p4mfe.normal import norm_cdf

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_STD_NORMAL = ContinuousDistribution(
    lambda z: np.exp(-0.5 * z * z) * _INV_SQRT_2PI
)


def risk_neutral_expectation(
    g: Callable,
    S0: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-9,
) -> float:
    """E_RN[g(S_T)] by integrating against phi; g must accept numpy arrays."""
    return _STD_NORMAL.expected_value(
        lambda z: g(price_from_normal(z, S0, T, sigma, r, q)), tol
    )


def risk_neutral_price(
    g: Callable,
    S0: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-9,
) -> float:
    """e^(-rT) E_RN[g(S_T)]."""
    return math.exp(-r * T) * risk_neutral_expectation(
        g, S0, T, sigma, r, q, tol
    )


def rn_expected_payoff(
    option_type: OptionType,
    S0: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    """Call: F N(d1) - K N(d2); put: K N(-d2) - F N(-d1); F = S_0 e^((r - q) T)."""
    d1, d2 = bs_d1_d2(S0, K, T, sigma, r, q)
    F = S0 * math.exp((r - q) * T)
    if option_type is OptionType.CALL:
        return F * norm_cdf(d1) - K * norm_cdf(d2)
    return K * norm_cdf(-d2) - F * norm_cdf(-d1)


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    std_error: float


def monte_carlo_price(
    g: Callable,
    S0: float,
    T: float,
    sigma: float,
    r: float,
    q: float,
    n: int,
    seed: int = 42,
    antithetic: bool = True,
) -> MonteCarloResult:
    """e^(-rT) mean of g(S_T) over n normal draws (pairs (z, -z) if antithetic),
    with standard error. g must accept numpy arrays."""
    if n < 2:
        raise ValueError("need n >= 2")
    z = np.random.default_rng(seed).standard_normal(n)
    x = g(price_from_normal(z, S0, T, sigma, r, q))
    if antithetic:
        x = 0.5 * (x + g(price_from_normal(-z, S0, T, sigma, r, q)))
    x = math.exp(-r * T) * np.asarray(x, float)
    return MonteCarloResult(
        float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))
    )


__all__ = [
    "MonteCarloResult",
    "OptionType",
    "monte_carlo_price",
    "risk_neutral_expectation",
    "risk_neutral_price",
    "rn_expected_payoff",
]
