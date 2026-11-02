"""Exercise 8: Gronwall's inequality. With W = M + int_0^x u v and
V = int_0^x v, g = W e^(-V) has g' = v (u - W) e^(-V) <= 0, so
u <= W <= M e^V.

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex08_gronwall.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Ex08Check:
    hypothesis_gap: float  # max (u - W): <= 0 when the hypothesis holds
    conclusion_gap: float  # max (u - M e^V), relative to M e^V
    g_increase: float  # max (g_{i+1} - g_i), relative to M


def ex08_check(
    u: Callable, v: Callable, m: float, x_end: float, n: int = 2000
) -> Ex08Check:
    x = np.linspace(0.0, x_end, n + 1)
    h = x_end / n
    ux, vx = np.array([u(t) for t in x]), np.array([v(t) for t in x])
    w_sum = m + np.concatenate(
        ([0.0], np.cumsum(0.5 * h * (ux[1:] * vx[1:] + ux[:-1] * vx[:-1])))
    )
    v_sum = np.concatenate(([0.0], np.cumsum(0.5 * h * (vx[1:] + vx[:-1]))))
    g = w_sum * np.exp(-v_sum)
    bound = m * np.exp(v_sum)
    return Ex08Check(
        float(np.max(ux - w_sum)),
        float(np.max((ux - bound) / bound)),
        float(np.max(np.diff(g)) / m),
    )


def ex08_integrand(t: float) -> float:
    return 1.0 + math.sin(t)


def ex08_integral(x: float) -> float:
    return x + 1.0 - math.cos(x)
