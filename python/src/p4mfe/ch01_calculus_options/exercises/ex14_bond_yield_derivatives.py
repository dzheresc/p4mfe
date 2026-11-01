"""Exercise 14: price-yield derivatives.

f(y) = sum c_i e^(-y t_i):           f' = -sum c_i t_i e^(-y t_i),  f'' = sum c_i t_i^2 e^(-y t_i).
g(y) = sum c_i (1 + y/m)^(-m t_i):   g' = -sum c_i t_i (1 + y/m)^(-m t_i - 1),
                                     g'' = sum c_i t_i (t_i + 1/m) (1 + y/m)^(-m t_i - 2).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PriceDerivs(NamedTuple):
    value: float
    first: float
    second: float


def ex14_continuous(times, amounts, y: float) -> PriceDerivs:
    t = np.asarray(times, dtype=float)
    df = np.asarray(amounts, dtype=float) * np.exp(-y * t)
    return PriceDerivs(float(df.sum()), float(-(t * df).sum()), float((t * t * df).sum()))


def ex14_discrete(times, amounts, y: float, m: float) -> PriceDerivs:
    if m <= 0:
        raise ValueError("m must be positive")
    t = np.asarray(times, dtype=float)
    c = np.asarray(amounts, dtype=float)
    base = 1.0 + y / m
    return PriceDerivs(
        float((c * base ** (-m * t)).sum()),
        float(-(c * t * base ** (-m * t - 1)).sum()),
        float((c * t * (t + 1.0 / m) * base ** (-m * t - 2)).sum()),
    )
