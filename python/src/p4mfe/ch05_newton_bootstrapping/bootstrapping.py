"""Section 5.5: bootstrapping for finding zero rate curves.

Chapter 2's bootstrap_zero_rates needs every coupon date to be an earlier
maturity. With sparser bonds, assume r(0, t) is linear between known
maturities: with the curve known up to T_last, the rates in (T_last, T] are
linear in the one unknown x = r(0, T),

    r(0, t) = r_last + (x - r_last) (t - T_last) / (T - T_last),

and pricing the new bond is one nonlinear equation in x, solved with Newton.
Before the first curve point the curve is flat; with no starting curve the
first bond is priced at a flat rate (its yield).
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from p4mfe.ch02_integration_bonds.bonds import BondQuote, CashFlows, ZeroCurve, bond_cash_flows

from .solvers import RootResult, Tolerances, newton

BOOTSTRAP_TOLERANCES = Tolerances(1e-12, 1e-12, 100)


def zero_rate_linear(curve: ZeroCurve, t: float) -> float:
    """Linear interpolation of the curve, flat outside it."""
    return float(np.interp(t, curve.times, curve.rates))


def bootstrap_next_rate(
    curve: ZeroCurve | None,
    cf: CashFlows,
    price: float,
    tol: Tolerances = BOOTSTRAP_TOLERANCES,
) -> RootResult:
    """r(0, T) at the maturity T of a bond, given the curve up to T_last < T."""
    maturity = float(cf.times[-1])
    empty = curve is None or len(curve.times) == 0
    last_maturity = 0.0 if empty else float(curve.times[-1])
    r_last = 0.0 if empty else float(curve.rates[-1])
    if not empty and not maturity > last_maturity:
        raise ValueError("maturity must exceed the curve")

    pv_known = 0.0
    c, t, w = [], [], []
    for ti, ci in zip(cf.times, cf.amounts):
        if not empty and ti <= last_maturity:
            pv_known += ci * math.exp(-zero_rate_linear(curve, ti) * ti)
        else:
            c.append(ci)
            t.append(ti)
            w.append(
                1.0
                if empty
                else (ti - last_maturity) / (maturity - last_maturity)
            )
    c, t, w = np.array(c), np.array(t), np.array(w)

    def disc(x):
        return c * np.exp(-(r_last + (x - r_last) * w) * t)

    return newton(
        lambda x: pv_known + float(np.sum(disc(x))) - price,
        lambda x: -float(np.sum(t * w * disc(x))),
        0.05 if empty else r_last,
        tol,
    )


def bootstrap_zero_curve(
    quotes: Iterable[BondQuote],
    freq: int,
    curve: ZeroCurve | None = None,
    face: float = 100.0,
    tol: Tolerances = BOOTSTRAP_TOLERANCES,
) -> ZeroCurve:
    """Adds one point (T, r(0, T)) per bond to `curve`, in order of maturity."""
    times = [] if curve is None else list(curve.times)
    rates = [] if curve is None else list(curve.rates)
    for q in sorted(quotes, key=lambda q: q.maturity):
        cf = bond_cash_flows(q.maturity, q.coupon_rate, freq, face)
        cur = ZeroCurve(np.array(times), np.array(rates)) if times else None
        rates.append(bootstrap_next_rate(cur, cf, q.price, tol).x)
        times.append(q.maturity)
    return ZeroCurve(np.array(times), np.array(rates))
