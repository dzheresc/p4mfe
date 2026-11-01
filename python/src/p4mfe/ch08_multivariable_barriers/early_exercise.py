"""Section 8.10: optimality of early exercise.

With no dividends, C >= S - K e^(-r tau) > S - K: an American call is never
exercised early and equals the European call. With dividends, or for puts,
early exercise has value. A Cox-Ross-Rubinstein tree prices both styles.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from p4mfe.ch01_calculus_options.options import OptionType, payoff
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call


class ExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


# 8.10 ----------------------------------------------------------------------
def binomial_price(option_type: OptionType, S: float, K: float, T: float, sigma: float, r: float, q: float, n: int,
                   style: ExerciseStyle) -> float:
    if n < 1 or not (T > 0 and sigma > 0):
        raise ValueError("need n >= 1, T, sigma > 0")
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("step too large (p not in (0, 1))")
    disc = math.exp(-r * dt)
    V = payoff(option_type, S * u ** (2.0 * np.arange(n + 1) - n), K)
    for i in range(n - 1, -1, -1):
        V = disc * (p * V[1:] + (1 - p) * V[:-1])
        if style is ExerciseStyle.AMERICAN:
            V = np.maximum(V, payoff(option_type, S * u ** (2.0 * np.arange(i + 1) - i), K))
    return float(V[0])


def early_exercise_premium(option_type: OptionType, S: float, K: float, T: float, sigma: float, r: float, q: float,
                           n: int = 500) -> float:
    return (binomial_price(option_type, S, K, T, sigma, r, q, n, ExerciseStyle.AMERICAN)
            - binomial_price(option_type, S, K, T, sigma, r, q, n, ExerciseStyle.EUROPEAN))


def call_exercise_loss(S: float, K: float, tau: float, sigma: float, r: float) -> float:
    """C - (S - K) for q = 0."""
    return bs_call(S, K, tau, sigma, r) - (S - K)


def call_exercise_loss_bound(K: float, tau: float, r: float) -> float:
    """K (1 - e^(-r tau))."""
    return -K * math.expm1(-r * tau)
