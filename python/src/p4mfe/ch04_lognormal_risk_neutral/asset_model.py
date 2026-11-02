"""Section 4.4: the lognormal model for asset prices.

    S_T = S_0 exp((mu - q - sigma^2/2) T + sigma sqrt(T) Z),  Z ~ N(0, 1),
so E[S_T] = S_0 e^((mu - q) T) and ln(S_T) is chapter 3's
``log_price_distribution``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from p4mfe.ch03_probability_black_scholes.normal_rv import (
    NormalParams,
    log_price_distribution,
    lognormal_cdf,
    lognormal_pdf,
)
from p4mfe.normal import norm_inv


def log_return_distribution(
    T: float, sigma: float, mu: float, q: float = 0.0
) -> NormalParams:
    """ln(S_T / S_0) ~ N((mu - q - sigma^2/2) T, sigma^2 T)."""
    return NormalParams((mu - q - 0.5 * sigma * sigma) * T, sigma * sigma * T)


def price_from_normal(
    z, S0: float, T: float, sigma: float, mu: float, q: float = 0.0
):
    """S_T for the standard normal draw z (accepts arrays)."""
    out = S0 * np.exp(
        (mu - q - 0.5 * sigma * sigma) * T
        + sigma * math.sqrt(T) * np.asarray(z, float)
    )
    return float(out) if np.ndim(out) == 0 else out


def price_mean(S0: float, T: float, mu: float, q: float = 0.0) -> float:
    return S0 * math.exp((mu - q) * T)


def price_variance(
    S0: float, T: float, sigma: float, mu: float, q: float = 0.0
) -> float:
    """S_0^2 e^(2 (mu - q) T) (e^(sigma^2 T) - 1)."""
    m = price_mean(S0, T, mu, q)
    return m * m * math.expm1(sigma * sigma * T)


def price_median(
    S0: float, T: float, sigma: float, mu: float, q: float = 0.0
) -> float:
    return price_from_normal(0.0, S0, T, sigma, mu, q)


def price_pdf(x, S0: float, T: float, sigma: float, mu: float, q: float = 0.0):
    L = log_price_distribution(S0, T, sigma, mu, q)
    return lognormal_pdf(x, L.mean, L.stddev)


def price_cdf(
    x: float, S0: float, T: float, sigma: float, mu: float, q: float = 0.0
) -> float:
    L = log_price_distribution(S0, T, sigma, mu, q)
    return lognormal_cdf(x, L.mean, L.stddev)


def price_quantile(
    p: float, S0: float, T: float, sigma: float, mu: float, q: float = 0.0
) -> float:
    if not 0 < p < 1:
        raise ValueError("need 0 < p < 1")
    return price_from_normal(norm_inv(p), S0, T, sigma, mu, q)


@dataclass(frozen=True)
class PriceRange:
    lo: float
    hi: float


def price_confidence_interval(
    confidence: float,
    S0: float,
    T: float,
    sigma: float,
    mu: float,
    q: float = 0.0,
) -> PriceRange:
    """[lo, hi] holding S_T with probability `confidence`, equal tails:
    S_0 exp((mu - q - sigma^2/2) T -/+ z sigma sqrt T), z = N^(-1)((1 + c)/2).
    """
    if not 0 < confidence < 1:
        raise ValueError("need 0 < confidence < 1")
    z = norm_inv(0.5 * (1 + confidence))
    return PriceRange(
        price_from_normal(-z, S0, T, sigma, mu, q),
        price_from_normal(z, S0, T, sigma, mu, q),
    )


__all__ = [
    "NormalParams",
    "PriceRange",
    "log_price_distribution",
    "log_return_distribution",
    "price_cdf",
    "price_confidence_interval",
    "price_from_normal",
    "price_mean",
    "price_median",
    "price_pdf",
    "price_quantile",
    "price_variance",
]
