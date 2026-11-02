"""Section 10.6: power series sum a_k (x - c)^k.

Radius of convergence R = 1 / limsup |a_k|^(1/k) (= lim |a_k / a_{k+1}| when
it exists). Inside (c - R, c + R) the series may be differentiated and
integrated term by term; same-center series multiply by the Cauchy product.
Series are truncated to their first coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch10_math_appendix.interpolation import polynomial_value


@dataclass
class PowerSeries:
    a: list[float]
    center: float = 0.0

    def __call__(self, x: float) -> float:
        return polynomial_value(self.a, x - self.center)


def derivative(s: PowerSeries) -> PowerSeries:
    return PowerSeries(
        [k * s.a[k] for k in range(1, len(s.a))] or [0.0], s.center
    )


def antiderivative(s: PowerSeries) -> PowerSeries:
    return PowerSeries(
        [0.0] + [c / (k + 1) for k, c in enumerate(s.a)], s.center
    )


def cauchy_product(p: PowerSeries, q: PowerSeries) -> PowerSeries:
    if p.center != q.center:
        raise ValueError("series need the same center")
    n = min(len(p.a), len(q.a))
    return PowerSeries(
        [sum(p.a[j] * q.a[k - j] for j in range(k + 1)) for k in range(n)],
        p.center,
    )


def radius_ratio_estimate(a) -> float:
    if len(a) < 2:
        raise ValueError("need two coefficients")
    return math.inf if a[-1] == 0 else abs(a[-2] / a[-1])


def radius_root_estimate(a) -> float:
    for n in range(len(a) - 1, 0, -1):
        if a[n] != 0:
            return abs(a[n]) ** (-1.0 / n)
    return math.inf


def exp_power_series(n: int) -> PowerSeries:
    return PowerSeries([1.0 / math.factorial(k) for k in range(n + 1)])


def geometric_power_series(n: int) -> PowerSeries:
    return PowerSeries([1.0] * (n + 1))


def log1p_power_series(n: int) -> PowerSeries:
    return PowerSeries(
        [0.0] + [(1.0 if k % 2 else -1.0) / k for k in range(1, n + 1)]
    )
