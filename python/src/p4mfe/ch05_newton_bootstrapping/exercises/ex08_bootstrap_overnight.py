"""Exercise 8: bootstrap from five semiannual bonds with overnight rate 5%, linear on the given intervals.

r(0, t) at 0, 0.5, 1, 5/3, 10/3, 5: 5%, 5.0636%, 4.9370%, 5.2983%, 4.5326%, 3.2119%.
The hint's r(0, 7/6) should read (3 r(0, 1) + r(0, 5/3))/4 (the book prints 4 r(0, 5/3)).

Full derivation: cpp/ch05_newton_bootstrapping/exercises/include/p4mfe/ch05/exercises/ex08_bootstrap_overnight.hpp
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch02_integration_bonds.bonds import BondQuote, ZeroCurve

from ..bootstrapping import bootstrap_zero_curve, zero_rate_linear

EX08_QUOTES = (
    BondQuote(0.5, 0.0, 97.5),
    BondQuote(1.0, 0.05, 100.0),
    BondQuote(5.0 / 3.0, 0.06, 103.0),
    BondQuote(10.0 / 3.0, 0.05, 102.0),
    BondQuote(5.0, 0.04, 103.0),
)
EX08_OVERNIGHT = 0.05


def ex08_curve() -> ZeroCurve:
    return bootstrap_zero_curve(
        EX08_QUOTES, 2, ZeroCurve(np.array([0.0]), np.array([EX08_OVERNIGHT]))
    )


def ex08_hint_residual(c: ZeroCurve) -> float:
    """Right side of (5.43) minus 103."""

    def r(t):
        return zero_rate_linear(c, t)

    x = r(5.0 / 3.0)
    r76 = (3.0 * r(1.0) + x) / 4.0
    return (
        3.0 * math.exp(-r(1.0 / 6.0) / 6.0)
        + 3.0 * math.exp(-2.0 * r(2.0 / 3.0) / 3.0)
        + 3.0 * math.exp(-7.0 * r76 / 6.0)
        + 103.0 * math.exp(-5.0 * x / 3.0)
        - 103.0
    )
