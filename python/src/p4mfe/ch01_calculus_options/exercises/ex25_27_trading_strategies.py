"""Exercises 25-27.

25. Same money in stock (S0 = 35) or ATM calls (4): R_S = S/35 - 1,
    R_C = (S - 35)^+ / 4 - 1; calls win iff S(T) > 35^2/31 = 39.516, so in the
    expected 40-45 range (25%..150% vs 14.3%..28.6%); below 35 calls lose 100%.
26. V = 2S - 3 (S - 20)^+ + 2 (S - 40)^+: long 2 shares, short 3 calls K = 20,
    long 2 calls K = 40 (cash V(0) = 0, first slope in shares, slope jumps in calls).
27. Range digital c 1{K1 <= S <= K2} ~ (c/eps)[C(K1) - C(K1+eps) - C(K2-eps) + C(K2)].
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .portfolio import Portfolio, call

EX25_SPOT = 35.0
EX25_CALL_PRICE = 4.0
EX25_CROSSOVER = EX25_SPOT**2 / (EX25_SPOT - EX25_CALL_PRICE)


def ex25_stock_return(spot_t):
    return spot_t / EX25_SPOT - 1.0


def ex25_call_return(spot_t):
    return np.maximum(spot_t - EX25_SPOT, 0.0) / EX25_CALL_PRICE - 1.0


def replicate_piecewise_linear(
    v0: float, slopes: Sequence[float], kinks: Sequence[float]
) -> Portfolio:
    """Cash v0 = V(0), shares = first slope, and (slope jump) calls at each kink."""
    if len(slopes) != len(kinks) + 1:
        raise ValueError("need one more slope than kinks")
    if any(b < a for a, b in zip(kinks, kinks[1:])):
        raise ValueError("kinks must be increasing")
    legs = [
        call(k, s1 - s0)
        for k, s0, s1 in zip(kinks, slopes, slopes[1:])
        if s1 != s0
    ]
    return Portfolio(cash=v0, shares=slopes[0], options=legs)


def ex26_target(spot_t):
    spot_t = np.asarray(spot_t, dtype=float)
    v = np.where(
        spot_t < 20, 2 * spot_t, np.where(spot_t < 40, 60 - spot_t, spot_t - 20)
    )
    return v if v.ndim else float(v)


def ex26_portfolio() -> Portfolio:
    return replicate_piecewise_linear(0.0, [2.0, -1.0, 1.0], [20.0, 40.0])


def ex27_target(spot_t, c: float, strike1: float, strike2: float):
    spot_t = np.asarray(spot_t, dtype=float)
    v = np.where((strike1 <= spot_t) & (spot_t <= strike2), c, 0.0)
    return v if v.ndim else float(v)


def ex27_range_digital(
    c: float, strike1: float, strike2: float, eps: float
) -> Portfolio:
    if not 0 < strike1 < strike2:
        raise ValueError("need 0 < K1 < K2")
    if not (eps > 0 and 2 * eps <= strike2 - strike1):
        raise ValueError("need 0 < 2 eps <= K2 - K1")
    n = c / eps
    return Portfolio(
        options=[
            call(strike1, n),
            call(strike1 + eps, -n),
            call(strike2 - eps, -n),
            call(strike2, n),
        ]
    )
