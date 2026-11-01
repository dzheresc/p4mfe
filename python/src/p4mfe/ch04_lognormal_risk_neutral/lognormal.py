"""Section 4.2: lognormal random variables, Y = e^X with X ~ N(mu, sigma^2).

The pdf, cdf, mean and variance are chapter 3's; this module adds moments of
any order, quantiles, median, mode and the fit from a mean and variance.
"""

from __future__ import annotations

import math

from p4mfe.ch03_probability_black_scholes.normal_rv import (
    NormalParams,
    lognormal_cdf,
    lognormal_mean,
    lognormal_pdf,
    lognormal_variance,
)
from p4mfe.normal import norm_inv


def lognormal_moment(n: float, mu: float, sigma: float) -> float:
    """E[Y^n] = E[e^(nX)] = e^(n mu + n^2 sigma^2 / 2)."""
    return math.exp(n * mu + 0.5 * n * n * sigma * sigma)


def lognormal_quantile(p: float, mu: float, sigma: float) -> float:
    """y_p = e^(mu + sigma N^(-1)(p))."""
    if not 0 < p < 1:
        raise ValueError("need 0 < p < 1")
    return math.exp(mu + sigma * norm_inv(p))


def lognormal_median(mu: float) -> float:
    return math.exp(mu)


def lognormal_mode(mu: float, sigma: float) -> float:
    """argmax of the density, e^(mu - sigma^2) < median < mean."""
    return math.exp(mu - sigma * sigma)


def lognormal_params_from_moments(m: float, v: float) -> NormalParams:
    """Parameters of ln(Y) for E[Y] = m, var(Y) = v: sigma^2 = ln(1 + v/m^2), mu = ln m - sigma^2/2."""
    if not (m > 0 and v > 0):
        raise ValueError("need m, v > 0")
    s2 = math.log1p(v / (m * m))
    return NormalParams(math.log(m) - 0.5 * s2, s2)


__all__ = [
    "NormalParams",
    "lognormal_cdf",
    "lognormal_mean",
    "lognormal_median",
    "lognormal_mode",
    "lognormal_moment",
    "lognormal_params_from_moments",
    "lognormal_pdf",
    "lognormal_quantile",
    "lognormal_variance",
]
