"""Exercises 7-8: confidence intervals for a lognormal price; American roulette.

7. S0 = 50, q = 2%, mu = 8%, sigma = 30%, r = 5%. With z = N^(-1)((1 + c)/2),
   S_T in S0 exp((mu - q - sigma^2/2) T -/+ z sigma sqrt(T)) with probability c; (ii) uses mu = r.
   Assumption: 15 days = 1/24 year, k months = k/12 year.
8. G = 200 W - 10000, W ~ Bin(100, 18/38): P(W >= 55) = 0.076689, P(W <= 45) = 0.354794 exactly;
   normal approximation 0.063202 and 0.317628 (0.076603 and 0.354127 with continuity correction).

Full derivations: cpp/ch04_lognormal_risk_neutral/exercises/include/p4mfe/ch04/exercises/ex07_08_confidence_roulette.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch04_lognormal_risk_neutral.asset_model import PriceRange, price_confidence_interval
from p4mfe.normal import norm_cdf


@dataclass(frozen=True)
class Ex07Asset:
    S0: float = 50.0
    q: float = 0.02
    mu: float = 0.08
    sigma: float = 0.3
    r: float = 0.05


EX07_HORIZONS = (1.0 / 24.0, 1.0 / 12.0, 2.0 / 12.0, 0.5, 1.0)


def ex07_interval(confidence: float, T: float, risk_neutral: bool, a: Ex07Asset = Ex07Asset()) -> PriceRange:
    return price_confidence_interval(confidence, a.S0, T, a.sigma, a.r if risk_neutral else a.mu, a.q)


EX08_P_BLACK = 18.0 / 38.0


def binomial_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * p ** k * (1.0 - p) ** (n - k)


@dataclass(frozen=True)
class RouletteOdds:
    win: float   # P(net gain >= threshold)
    lose: float  # P(net loss >= threshold)


def ex08_exact(n_bets: int = 100, bet: float = 100.0, threshold: float = 1000.0, p: float = EX08_P_BLACK) -> RouletteOdds:
    win = lose = 0.0
    for k in range(n_bets + 1):
        gain = bet * (2 * k - n_bets)
        if gain >= threshold:
            win += binomial_pmf(k, n_bets, p)
        if gain <= -threshold:
            lose += binomial_pmf(k, n_bets, p)
    return RouletteOdds(win, lose)


def ex08_normal(continuity: bool = False, n_bets: int = 100, bet: float = 100.0, threshold: float = 1000.0,
                p: float = EX08_P_BLACK) -> RouletteOdds:
    mean = n_bets * bet * (2.0 * p - 1.0)
    sd = bet * math.sqrt(n_bets * 4.0 * p * (1.0 - p))
    t = threshold - bet if continuity else threshold
    return RouletteOdds(norm_cdf((mean - t) / sd), norm_cdf((-t - mean) / sd))
