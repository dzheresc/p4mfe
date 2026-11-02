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
    strikes: tuple[float, float, float]
    prices: tuple[float, float, float]


def ex29_convexity_gap(x: ThreeCalls) -> float:
    strike1, strike2, strike3 = x.strikes
    if not strike1 < strike2 < strike3:
        raise ValueError("need K1 < K2 < K3")
    lam = (strike3 - strike2) / (strike3 - strike1)
    return lam * x.prices[0] + (1 - lam) * x.prices[2] - x.prices[1]


def ex29_no_arbitrage(x: ThreeCalls) -> bool:
    return (
        x.prices[2] >= 0
        and x.prices[1] >= x.prices[2]
        and ex29_convexity_gap(x) >= 0
    )


def ex29_arbitrage(x: ThreeCalls) -> Portfolio | None:
    (strike1, strike2, strike3), (call1, call2, call3) = x.strikes, x.prices
    if ex29_convexity_gap(x) < 0:
        w = strike3 - strike2
        return Portfolio(
            options=[
                call(strike1, 1.0, call1),
                call(strike2, -(strike3 - strike1) / w, call2),
                call(strike3, (strike2 - strike1) / w, call3),
            ]
        )
    if call2 < call3:
        return Portfolio(
            options=[call(strike2, 1.0, call2), call(strike3, -1.0, call3)]
        )
    if call3 < 0:
        return Portfolio(options=[call(strike3, 1.0, call3)])
    return None


def ex28_data() -> ThreeCalls:
    return ThreeCalls((100.0, 120.0, 130.0), (8.0, 5.0, 3.0))
