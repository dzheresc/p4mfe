"""Sections 3.3-3.4: the standard normal variable, normal and lognormal
random variables, and the lognormal model for stock prices.

phi and N themselves are ``p4mfe.normal.norm_pdf`` / ``norm_cdf``; section
3.9's approximations of N are in ``cdf_approx``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from p4mfe.normal import norm_cdf, norm_pdf

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# 3.3 -----------------------------------------------------------------------
def std_normal_mgf(t: float) -> float:
    """E[e^(tZ)] = e^(t^2/2)."""
    return math.exp(0.5 * t * t)


# 3.4 -----------------------------------------------------------------------
@dataclass(frozen=True)
class NormalParams:
    mean: float
    variance: float

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)


def normal_pdf(x, mu: float, sigma: float):
    """Density of N(mu, sigma^2); accepts numpy arrays."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = (np.asarray(x, float) - mu) / sigma
    out = np.exp(-0.5 * z * z) * _INV_SQRT_2PI / sigma
    return float(out) if np.ndim(out) == 0 else out


def normal_cdf(x: float, mu: float, sigma: float) -> float:
    """N((x - mu) / sigma)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return norm_cdf((x - mu) / sigma)


def linear_combination(
    a: float, X: NormalParams, b: float, Y: NormalParams, rho: float = 0.0
) -> NormalParams:
    """aX + bY for jointly normal X, Y with correlation rho."""
    return NormalParams(
        a * X.mean + b * Y.mean,
        a * a * X.variance
        + b * b * Y.variance
        + 2 * a * b * rho * X.stddev * Y.stddev,
    )


def lognormal_mean(mu: float, sigma: float) -> float:
    """E[e^X] = e^(mu + sigma^2/2), X ~ N(mu, sigma^2)."""
    return math.exp(mu + 0.5 * sigma * sigma)


def lognormal_variance(mu: float, sigma: float) -> float:
    """var(e^X) = e^(2 mu + sigma^2) (e^(sigma^2) - 1)."""
    s2 = sigma * sigma
    return math.exp(2 * mu + s2) * math.expm1(s2)


def lognormal_pdf(y, mu: float, sigma: float):
    """phi((ln y - mu)/sigma) / (sigma y) for y > 0, else 0; accepts arrays."""
    y = np.asarray(y, float)
    pos = y > 0
    safe = np.where(pos, y, 1.0)
    out = np.where(pos, normal_pdf(np.log(safe), mu, sigma) / safe, 0.0)
    return float(out) if np.ndim(out) == 0 else out


def lognormal_cdf(y: float, mu: float, sigma: float) -> float:
    return 0.0 if y <= 0 else normal_cdf(math.log(y), mu, sigma)


def log_price_distribution(
    spot0: float, maturity: float, sigma: float, mu: float, q: float = 0.0
) -> NormalParams:
    """ln(S_T) ~ N(ln S0 + (mu - q - sigma^2/2) T, sigma^2 T)."""
    return NormalParams(
        math.log(spot0) + (mu - q - 0.5 * sigma * sigma) * maturity,
        sigma * sigma * maturity,
    )


def prob_price_above(
    strike: float,
    spot0: float,
    maturity: float,
    sigma: float,
    mu: float,
    q: float = 0.0,
) -> float:
    """P(S_T > K); with mu = r it is N(d2)."""
    ln_spot_t = log_price_distribution(spot0, maturity, sigma, mu, q)
    return 1.0 - normal_cdf(math.log(strike), ln_spot_t.mean, ln_spot_t.stddev)


__all__ = [
    "NormalParams",
    "linear_combination",
    "log_price_distribution",
    "lognormal_cdf",
    "lognormal_mean",
    "lognormal_pdf",
    "lognormal_variance",
    "norm_cdf",
    "norm_pdf",
    "normal_cdf",
    "normal_pdf",
    "prob_price_above",
    "std_normal_mgf",
]
