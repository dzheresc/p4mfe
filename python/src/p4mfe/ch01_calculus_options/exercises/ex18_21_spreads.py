"""Exercises 18-21: spreads.

18. Butterfly 30/35/40: tent of height 5 at 35, zero outside (30, 40).
19. Bull call 30/35: clamp(S - 30, 0, 5).  Bear put 20/15: clamp(20 - S, 0, 5).
20. Butterfly 45/50/55 at 4, 6, 9 costs 1; P&L = payoff - 1, profitable for 46 < S(T) < 54.
21. Portfolio 1 (calls X-5 and X+5) minus Portfolio 2 (two calls X) is a butterfly >= 0,
    > 0 for |S(T) - X| < 5: prefer Portfolio 1 (it must also cost more).
"""

from __future__ import annotations

from .portfolio import Portfolio, call, put


def ex18_butterfly() -> Portfolio:
    return Portfolio(options=[call(30, 1), call(35, -2), call(40, 1)])


def ex19_bull_spread() -> Portfolio:
    return Portfolio(options=[call(30, 1), call(35, -1)])


def ex19_bear_spread() -> Portfolio:
    return Portfolio(options=[put(20, 1), put(15, -1)])


def ex20_butterfly() -> Portfolio:
    return Portfolio(
        options=[call(45, 1, 4.0), call(50, -2, 6.0), call(55, 1, 9.0)]
    )


EX20_BREAKEVEN_LOW = 46.0
EX20_BREAKEVEN_HIGH = 54.0


def ex21_portfolio1(x: float) -> Portfolio:
    return Portfolio(options=[call(x - 5, 1), call(x + 5, 1)])


def ex21_portfolio2(x: float) -> Portfolio:
    return Portfolio(options=[call(x, 2)])
