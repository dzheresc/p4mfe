"""Exercise 7: bootstrap Treasury prices with continuous and semiannual compounding.

(i) r(0, t) at 0.25, 0.5, 2, 3, 5, 10: 5.2341%, 5.0636%, 4.7289%, 4.7582%, 4.6303%, 4.6772%.
(ii) Same discount factors, (1 + r_2/2)^(-2t) = e^(-rt), so r_2 = 2(e^(r/2) - 1) > r: the semiannual curve is
     higher, by about r^2/4 (5.4-6.9 bp here).

Full derivation: cpp/ch05_newton_bootstrapping/exercises/include/p4mfe/ch05/exercises/ex07_treasury_bootstrap.hpp
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch02_integration_bonds.bonds import BondQuote, ZeroCurve, bond_cash_flows

from ..bootstrapping import bootstrap_zero_curve, zero_rate_linear
from ..solvers import Tolerances, newton

EX07_QUOTES = (
    BondQuote(0.25, 0.0, 98.7),
    BondQuote(0.5, 0.0, 97.5),
    BondQuote(2.0, 0.04875, 100.0 + 5.0 / 32.0),
    BondQuote(3.0, 0.04875, 100.0 + 5.0 / 32.0),
    BondQuote(5.0, 0.04625, 99.0 + 22.0 / 32.0),
    BondQuote(10.0, 0.04875, 101.0 + 4.0 / 32.0),
)


def ex07_curve_continuous(quotes=EX07_QUOTES) -> ZeroCurve:
    return bootstrap_zero_curve(quotes, 2)


def ex07_discount(r: float, t: float, m: int) -> float:
    """Discount factor for a rate compounded m times a year (m = 0: continuously)."""
    return math.exp(-r * t) if m == 0 else (1.0 + r / m) ** (-m * t)


def ex07_discount_dr(r: float, t: float, m: int) -> float:
    return (
        -t * math.exp(-r * t)
        if m == 0
        else -t * (1.0 + r / m) ** (-m * t - 1.0)
    )


def ex07_bootstrap_compounded(quotes, m: int, freq: int = 2) -> ZeroCurve:
    """Section 5.5's bootstrapping with m-compounded zero rates, linear between maturities."""
    times: list[float] = []
    rates: list[float] = []
    for bq in sorted(quotes, key=lambda q: q.maturity):
        cf = bond_cash_flows(bq.maturity, bq.coupon_rate, freq)
        T = bq.maturity
        empty = not times
        T_last = 0.0 if empty else times[-1]
        r_last = 0.0 if empty else rates[-1]
        curve = None if empty else ZeroCurve(np.array(times), np.array(rates))
        pv_known = 0.0
        c, t, w = [], [], []
        for ti, ci in zip(cf.times, cf.amounts):
            if not empty and ti <= T_last:
                pv_known += ci * ex07_discount(
                    zero_rate_linear(curve, ti), ti, m
                )
            else:
                c.append(ci)
                t.append(ti)
                w.append(1.0 if empty else (ti - T_last) / (T - T_last))

        def f(
            x, c=c, t=t, w=w, pv_known=pv_known, r_last=r_last, price=bq.price
        ):
            pv = sum(
                ci * ex07_discount(r_last + (x - r_last) * wi, ti, m)
                for ci, ti, wi in zip(c, t, w)
            )
            return pv_known + pv - price

        def df(x, c=c, t=t, w=w, r_last=r_last):
            return sum(
                ci * wi * ex07_discount_dr(r_last + (x - r_last) * wi, ti, m)
                for ci, ti, wi in zip(c, t, w)
            )

        rates.append(
            newton(
                f, df, 0.05 if empty else r_last, Tolerances(1e-12, 1e-12, 100)
            ).x
        )
        times.append(T)
    return ZeroCurve(np.array(times), np.array(rates))


def ex07_curve_semiannual(quotes=EX07_QUOTES) -> ZeroCurve:
    return ex07_bootstrap_compounded(quotes, 2)


def ex07_to_compounded(r: float, m: int) -> float:
    """m-compounded rate equivalent to a continuously compounded r."""
    return m * (math.exp(r / m) - 1.0)
