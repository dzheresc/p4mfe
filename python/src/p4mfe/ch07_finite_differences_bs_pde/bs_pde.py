"""Section 7.4: the Black-Scholes PDE

    dV/dt + 1/2 sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS - r V = 0.

7.4.1: a Delta-hedged short option replicates the payoff; the P&L has mean
~0 and standard deviation O(1/sqrt(n)) for n rebalancings, whatever the drift.
7.4.2: Theta + 1/2 sigma^2 S^2 Gamma + (r - q) S Delta - r V = 0 for any
portfolio of derivatives on one asset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from p4mfe.ch01_calculus_options.calculus import derivative, second_derivative
from p4mfe.ch01_calculus_options.options import payoff
from p4mfe.ch03_probability_black_scholes.black_scholes import Greeks, OptionType, bs_greeks, bs_price
from p4mfe.ch04_lognormal_risk_neutral.asset_model import price_from_normal


# 7.4 -----------------------------------------------------------------------
def bs_pde_residual_fd(
    V: Callable[[float, float], float],
    S: float,
    t: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    """PDE left-hand side for V(S, t), derivatives by central differences."""
    theta = derivative(lambda u: V(S, u), t)
    delta = derivative(lambda s: V(s, t), S)
    gamma = second_derivative(lambda s: V(s, t), S)
    return (
        theta
        + 0.5 * sigma * sigma * S * S * gamma
        + (r - q) * S * delta
        - r * V(S, t)
    )


def power_solution_rate(
    a: float, sigma: float, r: float, q: float = 0.0
) -> float:
    """c in V = S^a e^(c t): c = r - a (r - q) - 1/2 sigma^2 a (a - 1)."""
    return r - a * (r - q) - 0.5 * sigma * sigma * a * (a - 1)


def power_solution(
    a: float, S: float, t: float, sigma: float, r: float, q: float = 0.0
) -> float:
    return S**a * math.exp(power_solution_rate(a, sigma, r, q) * t)


def forward_contract_value(
    S: float, t: float, K: float, T: float, r: float, q: float = 0.0
) -> float:
    return S * math.exp(-q * (T - t)) - K * math.exp(-r * (T - t))


# 7.4.2 ---------------------------------------------------------------------
def pde_residual(
    V: float, g: Greeks, S: float, sigma: float, r: float, q: float = 0.0
) -> float:
    return (
        g.theta
        + 0.5 * sigma * sigma * S * S * g.gamma
        + (r - q) * S * g.delta
        - r * V
    )


def combine_greeks(w: Sequence[float], gs: Sequence[Greeks]) -> Greeks:
    if len(w) != len(gs):
        raise ValueError("size mismatch")
    return Greeks(
        delta=sum(wi * g.delta for wi, g in zip(w, gs)),
        gamma=sum(wi * g.gamma for wi, g in zip(w, gs)),
        vega=sum(wi * g.vega for wi, g in zip(w, gs)),
        theta=sum(wi * g.theta for wi, g in zip(w, gs)),
        rho=sum(wi * g.rho for wi, g in zip(w, gs)),
    )


# 7.4.1 ---------------------------------------------------------------------
def gbm_path(
    S0: float,
    T: float,
    sigma: float,
    mu: float,
    q: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """S_0, S_h, ..., S_T on n equal steps with real-world drift mu."""
    if n < 1:
        raise ValueError("need n >= 1")
    dt = T / n
    steps = price_from_normal(rng.standard_normal(n), 1.0, dt, sigma, mu, q)
    return S0 * np.concatenate(([1.0], np.cumprod(steps)))


def delta_hedge_pnl(
    option_type: OptionType,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float,
    path: Sequence[float],
) -> float:
    """Sell the option at its Black-Scholes price, hold Delta shares, cash at r,
    rebalance at each path point; return cash + shares S_T - payoff."""
    n = len(path) - 1
    if n < 1:
        raise ValueError("need at least two prices")
    dt = T / n
    growth = math.exp(r * dt)
    carry = math.exp(q * dt) - 1
    shares = bs_greeks(option_type, path[0], K, T, sigma, r, q).delta
    cash = bs_price(option_type, path[0], K, T, sigma, r, q) - shares * path[0]
    for k in range(1, n + 1):
        cash = cash * growth + shares * path[k - 1] * carry
        if k == n:
            break
        new_shares = bs_greeks(
            option_type, path[k], K, T - k * dt, sigma, r, q
        ).delta
        cash -= (new_shares - shares) * path[k]
        shares = new_shares
    return float(cash + shares * path[n] - payoff(option_type, path[n], K))


@dataclass(frozen=True)
class HedgeStats:
    mean: float
    std_dev: float


def delta_hedge_stats(
    option_type: OptionType,
    S0: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float,
    mu: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> HedgeStats:
    if n_paths < 2:
        raise ValueError("need n_paths >= 2")
    rng = np.random.default_rng(seed)
    pnl = np.array(
        [
            delta_hedge_pnl(
                option_type,
                K,
                T,
                sigma,
                r,
                q,
                gbm_path(S0, T, sigma, mu, q, n_steps, rng),
            )
            for _ in range(n_paths)
        ]
    )
    return HedgeStats(float(pnl.mean()), float(pnl.std(ddof=1)))
