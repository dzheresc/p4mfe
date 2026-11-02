"""Exercises 1-2: sums and averages of normal variables.

1. As written (X1 = Z, X2 = -Z) the sum is identically 0; the intended statement has X1, X2
   independent N(0, 1). Then the MGFs multiply, e^(t^2/2) e^(t^2/2) = e^(2 t^2/2), so
   X1 + X2 ~ N(0, 2), the law of sqrt(2) Z.
2. E[S_n] = mu by linearity; var(S_n) = (1/n^2)(n sigma^2 + 2 sum cov) = sigma^2/n when uncorrelated.

Full derivations: cpp/ch04_lognormal_risk_neutral/exercises/include/p4mfe/ch04/exercises/ex01_02_normal_averages.hpp
"""

from __future__ import annotations

from p4mfe.ch03_probability_black_scholes.normal_rv import NormalParams
from p4mfe.ch04_lognormal_risk_neutral.independent import independent_normal_combination, independent_normal_sum


def ex01_sum_independent() -> NormalParams:
    return independent_normal_sum(
        [NormalParams(0.0, 1.0), NormalParams(0.0, 1.0)]
    )


def ex01_sum_as_written(z: float) -> float:
    return z + (-z)


def ex02_average(mu: float, sigma: float, n: int) -> NormalParams:
    return independent_normal_combination(
        [1.0 / n] * n, [NormalParams(mu, sigma * sigma)] * n
    )
