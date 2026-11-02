"""Exercise 12: 1 + sigma^2 S^2/2 * Gamma/Theta is 0 if r = q = 0, and
1 / (1 + sigma N'(d2) / (2 r sqrt(tau) N(d2))) if q = 0, r != 0.

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex12_gamma_theta.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_d1_d2, bs_greeks
from p4mfe.normal import norm_cdf, norm_pdf


@dataclass(frozen=True)
class Ex12Market:
    S: float = 50.0
    K: float = 52.0
    tau: float = 0.75
    sigma: float = 0.3
    r: float = 0.05


def ex12_lhs(m: Ex12Market, option_type: OptionType = OptionType.CALL) -> float:
    g = bs_greeks(option_type, m.S, m.K, m.tau, m.sigma, m.r, 0.0)
    return 1.0 + 0.5 * m.sigma**2 * m.S**2 * g.gamma / g.theta


def ex12_rhs(m: Ex12Market) -> float:
    _, d2 = bs_d1_d2(m.S, m.K, m.tau, m.sigma, m.r, 0.0)
    return 1.0 / (
        1.0
        + m.sigma / (2.0 * m.r * math.sqrt(m.tau)) * norm_pdf(d2) / norm_cdf(d2)
    )
