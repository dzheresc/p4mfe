"""Exercises 5-6: differentiating integrals with variable limits.

Exercise 5. g(x) = N(b(x)), b(x) = (ln(x/K) + (r + sigma^2/2) T)/(sigma sqrt T):
    g'(x) = phi(b(x)) b'(x) = e^(-b(x)^2/2) / (x sigma sqrt(2 pi T))   (the call's Gamma).

Exercise 6. g(t) = int_t^inf (x - t) h(x) dx = int_t^inf x h - t int_t^inf h, so
    g'(t) = -t h(t) - int_t^inf h + t h(t) = -int_t^inf h(x) dx,   g''(t) = h(t).
    With h = e^(-rT) f (f the density of S(T)), C(K) = int_K^inf (x-K) h(x) dx gives
    d2C/dK2 = e^(-rT) f(K).

Full derivations: cpp/ch02_integration_bonds/exercises/include/p4mfe/ch02/exercises/ex05_06_integral_derivatives.hpp
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import numpy as np

from p4mfe.ch02_integration_bonds.integration import improper_integral
from p4mfe.normal import norm_cdf, norm_pdf


class Ex05Params(NamedTuple):
    strike: float
    maturity: float
    sigma: float
    r: float


def ex05_b(p: Ex05Params, x: float) -> float:
    return (
        math.log(x / p.strike) + (p.r + 0.5 * p.sigma**2) * p.maturity
    ) / (p.sigma * math.sqrt(p.maturity))


def ex05_g(p: Ex05Params, x: float) -> float:
    return norm_cdf(ex05_b(p, x))


def ex05_g_prime(p: Ex05Params, x: float) -> float:
    b = ex05_b(p, x)
    return math.exp(-0.5 * b * b) / (
        x * p.sigma * math.sqrt(2 * math.pi * p.maturity)
    )


def ex06_g_normal(t: float) -> float:
    """g(t) for h = phi: phi(t) - t (1 - N(t))."""
    return norm_pdf(t) - t * norm_cdf(-t)


def ex06_g_normal_prime(t: float) -> float:
    return -norm_cdf(-t)


def ex06_g(h: Callable, t: float, tol: float = 1e-10) -> float:
    """g(t) = int_t^inf (x - t) h(x) dx numerically (h must accept numpy arrays)."""
    return improper_integral(lambda x: (x - t) * h(x), t, math.inf, tol).value


class Ex06Market(NamedTuple):
    spot: float
    maturity: float
    sigma: float
    r: float


def ex06_call(m: Ex06Market, strike: float) -> float:
    sv = m.sigma * math.sqrt(m.maturity)
    d1 = (
        math.log(m.spot / strike) + (m.r + 0.5 * m.sigma**2) * m.maturity
    ) / sv
    return m.spot * norm_cdf(d1) - strike * math.exp(
        -m.r * m.maturity
    ) * norm_cdf(d1 - sv)


def ex06_discounted_density(m: Ex06Market, strike):
    """e^(-rT) times the lognormal density of S(T) at K (accepts arrays)."""
    strike = np.asarray(strike, dtype=float)
    sv = m.sigma * math.sqrt(m.maturity)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (
            np.log(strike / m.spot) - (m.r - 0.5 * m.sigma**2) * m.maturity
        ) / sv
        out = np.where(
            strike > 0,
            math.exp(-m.r * m.maturity)
            * np.exp(-0.5 * z * z)
            / (strike * sv * math.sqrt(2 * math.pi)),
            0.0,
        )
    return float(out) if out.ndim == 0 else out
