"""Exercises 25-29: Delta, Delta-Gamma and Delta-Gamma-vega neutral hedges.

25. Delta 5000, Gamma -200; call (0.4, 0.05), put (-0.5, 0.07). Options only: sell 4716.98
    calls and buy 6226.42 puts. With the asset: buy 4000 calls and sell 6600 shares (or buy
    2857.14 puts and sell 3571.43 shares).
26. Delta(C) = 0.653501 (S = 92, K = 90, T = 0.25, sigma = 0.2, r = 0.05): short 653.5 shares.
27. (i) 9634.88; (ii) short 588.59 shares; (iii) S = 98 a trading day later: options + shares
    = -49228.61 (from -49224.03), +5.19 with the cash; (iv) unhedged loss 1181.75.
28. (i) 22927.35; (ii) Delta -402.77: buy 402.77 shares (hold 802.77);
    (iii) S = 24 a month later: value 23399.34, sell 291.64 shares (hold 511.13).
29. Sell 670 calls, sell 110 puts, sell 254 shares.

Full derivations: cpp/ch03_probability_black_scholes/exercises/include/p4mfe/ch03/exercises/ex25_29_hedging.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_greeks, bs_put
from p4mfe.ch03_probability_black_scholes.hedging import delta_hedge, delta_hedge_shares


@dataclass(frozen=True)
class TwoOptionHedge:
    calls: float
    puts: float


def ex25_options_only(
    port_delta: float,
    port_gamma: float,
    dc: float,
    gc: float,
    dp: float,
    gp: float,
) -> TwoOptionHedge:
    det = dc * gp - dp * gc
    if det == 0:
        raise ValueError("the two options have proportional (Delta, Gamma)")
    return TwoOptionHedge(
        (-port_delta * gp + dp * port_gamma) / det,
        (-dc * port_gamma + gc * port_delta) / det,
    )


def ex26_shares() -> float:
    return delta_hedge_shares(
        1000 * bs_greeks(OptionType.CALL, 92.0, 90.0, 0.25, 0.2, 0.05).delta
    )


@dataclass(frozen=True)
class Ex27Result:
    premium: float
    shares: float
    cash: float
    position_value: float
    portfolio_value: float
    unhedged_loss: float


def ex27(dt: float = 1 / 252, spot1: float = 98.0) -> Ex27Result:
    n, spot, strike, maturity, sigma, r = 1000.0, 100.0, 100.0, 0.5, 0.3, 0.05
    call0 = bs_call(spot, strike, maturity, sigma, r)
    pos = delta_hedge(
        n,
        call0,
        bs_greeks(OptionType.CALL, spot, strike, maturity, sigma, r).delta,
        spot,
    )
    call1 = bs_call(spot1, strike, maturity - dt, sigma, r)
    return Ex27Result(
        n * call0,
        pos.shares,
        pos.cash,
        n * call1 + pos.shares * spot1,
        pos.value(call1, spot1, r, dt),
        n * (call0 - call1),
    )


@dataclass(frozen=True)
class Ex28Result:
    value0: float
    port_delta0: float
    shares1: float
    cash1: float
    value1: float
    shares2: float


def ex28() -> Ex28Result:
    n, strike, maturity, sigma, r, spot0, spot1, dt = (
        1000.0,
        25.0,
        0.5,
        0.3,
        0.04,
        20.0,
        24.0,
        1 / 12,
    )
    shares0, cash0 = 400.0, 10000.0
    put0 = bs_put(spot0, strike, maturity, sigma, r)
    delta0 = bs_greeks(OptionType.PUT, spot0, strike, maturity, sigma, r).delta
    value0 = n * put0 + shares0 * spot0 + cash0
    port_delta0 = n * delta0 + shares0
    shares1 = shares0 + delta_hedge_shares(port_delta0)
    cash1 = cash0 - (shares1 - shares0) * spot0
    put1 = bs_put(spot1, strike, maturity - dt, sigma, r)
    delta1 = bs_greeks(
        OptionType.PUT, spot1, strike, maturity - dt, sigma, r
    ).delta
    value1 = n * put1 + shares1 * spot1 + cash1 * math.exp(r * dt)
    return Ex28Result(value0, port_delta0, shares1, cash1, value1, -n * delta1)


@dataclass(frozen=True)
class Ex29Hedge:
    calls: float
    puts: float
    shares: float


def ex29_hedge(
    pd: float = 300.0, pg: float = 100.0, pv: float = 89.0
) -> Ex29Hedge:
    cd, cg, cv, qd, qg, qv = 0.2, 0.1, 0.1, -0.8, 0.3, 0.2
    det = cg * qv - qg * cv
    x = (-pg * qv + qg * pv) / det
    y = (-cg * pv + cv * pg) / det
    return Ex29Hedge(x, y, -(pd + x * cd + y * qd))
