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
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> tuple[float, float]:
    """d1 = (ln(S/K) + (r - q + sigma^2/2) T) / (sigma sqrt T), d2 = d1 - sigma sqrt T."""
    if not (spot > 0 and strike > 0 and maturity > 0 and sigma > 0):
        raise ValueError("need S, K, T, sigma > 0")
    sd = sigma * math.sqrt(maturity)
    d1 = (
        math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * maturity
    ) / sd
    return d1, d1 - sd


def bs_price(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    """C = S e^(-qT) N(d1) - K e^(-rT) N(d2);  P = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)."""
    d1, d2 = bs_d1_d2(spot, strike, maturity, sigma, r, q)
    fwd_spot = spot * math.exp(-q * maturity)
    disc_strike = strike * math.exp(-r * maturity)
    if option_type is OptionType.CALL:
        return fwd_spot * norm_cdf(d1) - disc_strike * norm_cdf(d2)
    return disc_strike * norm_cdf(-d2) - fwd_spot * norm_cdf(-d1)


def bs_call(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return bs_price(OptionType.CALL, spot, strike, maturity, sigma, r, q)


def bs_put(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return bs_price(OptionType.PUT, spot, strike, maturity, sigma, r, q)


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
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> Greeks:
    """Closed-form Greeks; see the C++ header for the formulas."""
    d1, d2 = bs_d1_d2(spot, strike, maturity, sigma, r, q)
    eq, er, sqrt_maturity = (
        math.exp(-q * maturity),
        math.exp(-r * maturity),
        math.sqrt(maturity),
    )
    pdf1 = norm_pdf(d1)
    gamma = eq * pdf1 / (spot * sigma * sqrt_maturity)
    vega = spot * eq * pdf1 * sqrt_maturity
    decay = -spot * sigma * eq * pdf1 / (2 * sqrt_maturity)
    if option_type is OptionType.CALL:
        delta = eq * norm_cdf(d1)
        theta = (
            decay
            + q * spot * eq * norm_cdf(d1)
            - r * strike * er * norm_cdf(d2)
        )
        rho = strike * maturity * er * norm_cdf(d2)
    else:
        delta = -eq * norm_cdf(-d1)
        theta = (
            decay
            - q * spot * eq * norm_cdf(-d1)
            + r * strike * er * norm_cdf(-d2)
        )
        rho = -strike * maturity * er * norm_cdf(-d2)
    return Greeks(delta, gamma, vega, theta, rho)


# 3.6.1 ---------------------------------------------------------------------
def bs_phi_identity_gap(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    """S e^(-qT) phi(d1) - K e^(-rT) phi(d2), zero up to rounding; the identity
    that makes the d(d1)/dS and d(d2)/dS terms cancel in every Greek."""
    d1, d2 = bs_d1_d2(spot, strike, maturity, sigma, r, q)
    return spot * math.exp(-q * maturity) * norm_pdf(d1) - strike * math.exp(
        -r * maturity
    ) * norm_pdf(d2)


def bs_pde_residual(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    """Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V (zero)."""
    g = bs_greeks(option_type, spot, strike, maturity, sigma, r, q)
    value = bs_price(option_type, spot, strike, maturity, sigma, r, q)
    return (
        g.theta
        + 0.5 * sigma * sigma * spot * spot * g.gamma
        + (r - q) * spot * g.delta
        - r * value
    )
