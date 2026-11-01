"""Exercises 28-29: convexity of call prices in the strike.

29. With lambda = (K3 - K2) / (K3 - K1), no arbitrage with the three calls iff
    C3 >= 0, C2 >= C3 and C2 <= lambda C1 + (1 - lambda) C3 (the three edges of
    the cone of nonnegative payoffs: long C3, bull spread, butterfly).
28. K = 100, 120, 130 at 8, 5, 3: 8/3 + 2 = 4.667 < 5. Buy 1 C100, sell 3 C120,
    buy 2 C130: +$1 now, payoff >= 0.
"""

from __future__ import annotations

from typing import NamedTuple

from .portfolio import Portfolio, call


class ThreeCalls(NamedTuple):
    K: tuple[float, float, float]
    C: tuple[float, float, float]


def ex29_convexity_gap(x: ThreeCalls) -> float:
    K1, K2, K3 = x.K
    if not K1 < K2 < K3:
        raise ValueError("need K1 < K2 < K3")
    lam = (K3 - K2) / (K3 - K1)
    return lam * x.C[0] + (1 - lam) * x.C[2] - x.C[1]


def ex29_no_arbitrage(x: ThreeCalls) -> bool:
    return x.C[2] >= 0 and x.C[1] >= x.C[2] and ex29_convexity_gap(x) >= 0


def ex29_arbitrage(x: ThreeCalls) -> Portfolio | None:
    (K1, K2, K3), (C1, C2, C3) = x.K, x.C
    if ex29_convexity_gap(x) < 0:
        w = K3 - K2
        return Portfolio(options=[call(K1, 1.0, C1), call(K2, -(K3 - K1) / w, C2), call(K3, (K2 - K1) / w, C3)])
    if C2 < C3:
        return Portfolio(options=[call(K2, 1.0, C2), call(K3, -1.0, C3)])
    if C3 < 0:
        return Portfolio(options=[call(K3, 1.0, C3)])
    return None


def ex28_data() -> ThreeCalls:
    return ThreeCalls((100.0, 120.0, 130.0), (8.0, 5.0, 3.0))
