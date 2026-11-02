"""Sections 6.8-6.8.3: ATM approximations of the Black-Scholes formulas.

    C ~ sigma S sqrt(T/(2 pi)) (1 - (r+q) T/2) + (r-q) T S/2
    P ~ sigma S sqrt(T/(2 pi)) (1 - (r+q) T/2) - (r-q) T S/2
    r = q = 0:  C = P ~ sigma S sqrt(T/(2 pi)) ~ 0.4 sigma S sqrt(T)
Derived by N(x) ~ 1/2 + x/sqrt(2 pi). For r = q = 0 the approximation
overestimates, with relative error (to the approximation) at most sigma^2 T / 24.
"""

from __future__ import annotations

import math

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch03_probability_black_scholes.black_scholes import Greeks, bs_d1_d2
from p4mfe.normal import norm_cdf

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# 6.8.1 ---------------------------------------------------------------------
def atm_price_approx(spot: float, maturity: float, sigma: float) -> float:
    return sigma * spot * math.sqrt(maturity) * _INV_SQRT_2PI


def atm_call_approx(
    spot: float, maturity: float, sigma: float, r: float, q: float = 0.0
) -> float:
    return (
        atm_price_approx(spot, maturity, sigma) * (1 - 0.5 * (r + q) * maturity)
        + 0.5 * (r - q) * maturity * spot
    )


def atm_put_approx(
    spot: float, maturity: float, sigma: float, r: float, q: float = 0.0
) -> float:
    return (
        atm_price_approx(spot, maturity, sigma) * (1 - 0.5 * (r + q) * maturity)
        - 0.5 * (r - q) * maturity * spot
    )


def atm_implied_vol_approx(price: float, spot: float, maturity: float) -> float:
    if not (spot > 0 and maturity > 0):
        raise ValueError("need S, T > 0")
    return price / (spot * math.sqrt(maturity) * _INV_SQRT_2PI)


def atm_greeks_approx(
    option_type: OptionType, spot: float, maturity: float, sigma: float
) -> Greeks:
    """First order in sigma sqrt(T), r = q = 0; see the C++ header."""
    sigma_sqrt_t = sigma * math.sqrt(maturity)
    half_d = 0.5 * sigma_sqrt_t * _INV_SQRT_2PI
    delta = 0.5 + half_d
    rho = spot * maturity * (0.5 - half_d)
    if option_type is OptionType.PUT:
        delta -= 1
        rho -= spot * maturity
    return Greeks(
        delta=delta,
        gamma=_INV_SQRT_2PI / (spot * sigma_sqrt_t),
        vega=spot * math.sqrt(maturity) * _INV_SQRT_2PI,
        theta=-0.5 * spot * sigma * _INV_SQRT_2PI / math.sqrt(maturity),
        rho=rho,
    )


# 6.8.2 ---------------------------------------------------------------------
def norm_cdf_linear(x: float) -> float:
    return 0.5 + x * _INV_SQRT_2PI


def atm_call_linearized(
    spot: float, maturity: float, sigma: float, r: float, q: float = 0.0
) -> float:
    d1, d2 = bs_d1_d2(spot, spot, maturity, sigma, r, q)
    return spot * math.exp(-q * maturity) * norm_cdf_linear(
        d1
    ) - spot * math.exp(-r * maturity) * norm_cdf_linear(d2)


# 6.8.3 ---------------------------------------------------------------------
def atm_price_exact(spot: float, maturity: float, sigma: float) -> float:
    x = 0.5 * sigma * math.sqrt(maturity)
    return spot * (norm_cdf(x) - norm_cdf(-x))


def atm_relative_error(maturity: float, sigma: float) -> float:
    a = atm_price_approx(1.0, maturity, sigma)
    return (a - atm_price_exact(1.0, maturity, sigma)) / a


def atm_relative_error_bound(maturity: float, sigma: float) -> float:
    return sigma * sigma * maturity / 24
