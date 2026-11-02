"""Section 8.10: optimality of early exercise.

With no dividends, C >= S - K e^(-r tau) > S - K: an American call is never
exercised early and equals the European call. With dividends, or for puts,
early exercise has value. A Cox-Ross-Rubinstein tree prices both styles.
"""

from __future__ import annotations

import math
import sys
from enum import Enum

import numpy as np

from p4mfe.ch01_calculus_options.options import OptionType, payoff
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call


class ExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


_TINY = sys.float_info.min


# 8.10 ----------------------------------------------------------------------
def binomial_price(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float,
    n: int,
    style: ExerciseStyle,
) -> float:
    if n < 1 or not (T > 0 and sigma > 0):
        raise ValueError("need n >= 1, T, sigma > 0")
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("step too large (p not in (0, 1))")
    disc = math.exp(-r * dt)
    pu = disc * p
    pd = disc * (1 - p)
    # spot[n + k] = S u^k for k = -n..n; row i of the tree uses k = -i, -i+2, ..., i.
    spot = S * u ** np.arange(-n, n + 1, dtype=float)
    v = payoff(option_type, spot[::2], K)
    # Values far out of the money decay geometrically down the tree and reach
    # the subnormal range after about a thousand steps, where each operation
    # is very slow on Intel CPUs. Values below the smallest normal double
    # (about 1e-308, far below any price) are set to 0 instead.
    if style is ExerciseStyle.AMERICAN:
        for i in range(n - 1, -1, -1):
            v = pu * v[1:] + pd * v[:-1]
            v[v < _TINY] = 0.0
            np.maximum(
                v, payoff(option_type, spot[n - i : n + i + 1 : 2], K), out=v
            )
        return float(v[0])
    # European: only nodes in [lo, hi] are nonzero (a put's zeros are at the
    # top of the tree, a call's at the bottom). The range widens by at most one
    # node per step and is trimmed once its end values drop below _TINY.
    nonzero = np.flatnonzero(v)
    if nonzero.size == 0:
        return 0.0
    lo, hi = int(nonzero[0]), int(nonzero[-1])
    for i in range(n - 1, -1, -1):
        lo = max(lo - 1, 0)
        hi = min(hi, i)
        v[lo : hi + 1] = pu * v[lo + 1 : hi + 2] + pd * v[lo : hi + 1]
        while hi > lo and v[hi] < _TINY:
            v[hi] = 0.0
            hi -= 1
        while lo < hi and v[lo] < _TINY:
            v[lo] = 0.0
            lo += 1
        if v[lo] < _TINY:  # lo == hi and the last value vanished.
            return 0.0
    return float(v[0])


def early_exercise_premium(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float,
    n: int = 500,
) -> float:
    return binomial_price(
        option_type, S, K, T, sigma, r, q, n, ExerciseStyle.AMERICAN
    ) - binomial_price(
        option_type, S, K, T, sigma, r, q, n, ExerciseStyle.EUROPEAN
    )


def call_exercise_loss(
    S: float, K: float, tau: float, sigma: float, r: float
) -> float:
    """C - (S - K) for q = 0."""
    return bs_call(S, K, tau, sigma, r) - (S - K)


def call_exercise_loss_bound(K: float, tau: float, r: float) -> float:
    """K (1 - e^(-r tau))."""
    return -K * math.expm1(-r * tau)
