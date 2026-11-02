"""Sections 3.5-3.6.1: Black-Scholes prices and Greeks with continuous
dividend yield q.

Argument order follows the book: S, K, T, sigma, r, q. Theta is dV/dt in
calendar time (minus dV/dT), per year; vega and rho are per unit change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.normal import norm_cdf, norm_pdf

__all__ = [
    "Greeks",
    "OptionType",
    "bs_call",
    "bs_d1_d2",
    "bs_greeks",
    "bs_pde_residual",
    "bs_phi_identity_gap",
    "bs_price",
    "bs_put",
]


# 3.5 -----------------------------------------------------------------------
def bs_d1_d2(
    S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0
) -> tuple[float, float]:
    """d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt T), d2 = d1 - sigma sqrt T."""
    if not (S > 0 and K > 0 and T > 0 and sigma > 0):
        raise ValueError("need S, K, T, sigma > 0")
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    return d1, d1 - sd


def bs_price(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    """C = S e^(-qT) N(d1) - K e^(-rT) N(d2);  P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)."""
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    fwd_S = S * math.exp(-q * T)
    disc_K = K * math.exp(-r * T)
    if option_type is OptionType.CALL:
        return fwd_S * norm_cdf(d1) - disc_K * norm_cdf(d2)
    return disc_K * norm_cdf(-d2) - fwd_S * norm_cdf(-d1)


def bs_call(
    S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0
) -> float:
    return bs_price(OptionType.CALL, S, K, T, sigma, r, q)


def bs_put(
    S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0
) -> float:
    return bs_price(OptionType.PUT, S, K, T, sigma, r, q)


# 3.6 -----------------------------------------------------------------------
@dataclass(frozen=True)
class Greeks:
    delta: float  # dV/dS
    gamma: float  # d2V/dS2
    vega: float  # dV/dsigma
    theta: float  # dV/dt
    rho: float  # dV/dr


def bs_greeks(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> Greeks:
    """Closed-form Greeks; see the C++ header for the formulas."""
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    eq, er, sqrtT = math.exp(-q * T), math.exp(-r * T), math.sqrt(T)
    pdf1 = norm_pdf(d1)
    gamma = eq * pdf1 / (S * sigma * sqrtT)
    vega = S * eq * pdf1 * sqrtT
    decay = -S * sigma * eq * pdf1 / (2 * sqrtT)
    if option_type is OptionType.CALL:
        delta = eq * norm_cdf(d1)
        theta = decay + q * S * eq * norm_cdf(d1) - r * K * er * norm_cdf(d2)
        rho = K * T * er * norm_cdf(d2)
    else:
        delta = -eq * norm_cdf(-d1)
        theta = decay - q * S * eq * norm_cdf(-d1) + r * K * er * norm_cdf(-d2)
        rho = -K * T * er * norm_cdf(-d2)
    return Greeks(delta, gamma, vega, theta, rho)


# 3.6.1 ---------------------------------------------------------------------
def bs_phi_identity_gap(
    S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0
) -> float:
    """S e^(-qT) phi(d1) - K e^(-rT) phi(d2), zero up to rounding; the identity
    that makes the d(d1)/dS and d(d2)/dS terms cancel in every Greek."""
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    return S * math.exp(-q * T) * norm_pdf(d1) - K * math.exp(
        -r * T
    ) * norm_pdf(d2)


def bs_pde_residual(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    """Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V (zero)."""
    g = bs_greeks(option_type, S, K, T, sigma, r, q)
    V = bs_price(option_type, S, K, T, sigma, r, q)
    return (
        g.theta
        + 0.5 * sigma * sigma * S * S * g.gamma
        + (r - q) * S * g.delta
        - r * V
    )
