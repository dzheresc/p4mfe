"""Sections 9.2-9.4: optimal investment portfolios.

Mean mu . w, variance w^T Sigma w, sum w = 1. With A = 1^T Sigma^-1 1,
B = 1^T Sigma^-1 mu, C = mu^T Sigma^-1 mu, D = A C - B^2:
    9.3  w = Sigma^-1 ((C - B mu_P) 1 + (A mu_P - B) mu) / D,
         sigma_P^2 = (A mu_P^2 - 2 B mu_P + C) / D
         with cash: w_risky = (mu_P - r_f) Sigma^-1 m / (m^T Sigma^-1 m), m = mu - r_f
    9.4  mu_P = (B + sqrt(D (A sigma_P^2 - 1))) / A;  with cash
         mu_P = r_f + sigma_P sqrt(m^T Sigma^-1 m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from p4mfe.ch05_newton_bootstrapping.newton_nd import solve_linear


# 9.2 -----------------------------------------------------------------------
def covariance_matrix(sigma, corr) -> np.ndarray:
    s = np.asarray(sigma, float)
    return np.asarray(corr, float) * np.outer(s, s)


def portfolio_mean(w, mu) -> float:
    return float(np.dot(w, mu))


def portfolio_variance(w, Sigma) -> float:
    w = np.asarray(w, float)
    return float(w @ np.asarray(Sigma, float) @ w)


def portfolio_std_dev(w, Sigma) -> float:
    return math.sqrt(portfolio_variance(w, Sigma))


@dataclass(frozen=True)
class FrontierConstants:
    A: float
    B: float
    C: float
    D: float


def frontier_constants(mu, Sigma) -> FrontierConstants:
    mu = np.asarray(mu, float)
    ones = np.ones_like(mu)
    Si1, Simu = solve_linear(Sigma, ones), solve_linear(Sigma, mu)
    A, B, C = float(ones @ Si1), float(ones @ Simu), float(mu @ Simu)
    D = A * C - B * B
    if not D > 0:
        raise ValueError("need non-constant expected returns")
    return FrontierConstants(A, B, C, D)


# 9.3 -----------------------------------------------------------------------
def global_min_variance_portfolio(Sigma) -> np.ndarray:
    w = solve_linear(Sigma, np.ones(len(Sigma)))
    return w / w.sum()


def min_variance_portfolio(mu, Sigma, target_mean: float) -> np.ndarray:
    c = frontier_constants(mu, Sigma)
    rhs = (
        (c.C - c.B * target_mean)
        + (c.A * target_mean - c.B) * np.asarray(mu, float)
    ) / c.D
    return solve_linear(Sigma, rhs)


def frontier_variance(mu, Sigma, target_mean: float) -> float:
    c = frontier_constants(mu, Sigma)
    return (c.A * target_mean**2 - 2 * c.B * target_mean + c.C) / c.D


def min_variance_portfolio_with_cash(
    mu, Sigma, r_f: float, target_mean: float
) -> np.ndarray:
    """Risky weights; the cash weight is 1 - sum(w)."""
    m = np.asarray(mu, float) - r_f
    w = solve_linear(Sigma, m)
    return w * (target_mean - r_f) / float(m @ w)


def tangency_portfolio(mu, Sigma, r_f: float) -> np.ndarray:
    w = solve_linear(Sigma, np.asarray(mu, float) - r_f)
    if w.sum() == 0:
        raise ValueError("undefined (r_f equals the GMV mean)")
    return w / w.sum()


def sharpe_ratio(w, mu, Sigma, r_f: float) -> float:
    return (portfolio_mean(w, mu) - r_f) / portfolio_std_dev(w, Sigma)


# 9.4 -----------------------------------------------------------------------
def max_return_for_std_dev(mu, Sigma, sigma_P: float) -> float:
    c = frontier_constants(mu, Sigma)
    disc = c.A * sigma_P**2 - 1
    if disc < 0:
        raise ValueError("sigma_P below the minimum 1/sqrt(A)")
    return (c.B + math.sqrt(c.D * disc)) / c.A


def max_return_portfolio(mu, Sigma, sigma_P: float) -> np.ndarray:
    return min_variance_portfolio(
        mu, Sigma, max_return_for_std_dev(mu, Sigma, sigma_P)
    )


def max_return_with_cash(mu, Sigma, r_f: float, sigma_P: float) -> float:
    m = np.asarray(mu, float) - r_f
    return r_f + sigma_P * math.sqrt(float(m @ solve_linear(Sigma, m)))


def max_return_portfolio_with_cash(
    mu, Sigma, r_f: float, sigma_P: float
) -> np.ndarray:
    return min_variance_portfolio_with_cash(
        mu, Sigma, r_f, max_return_with_cash(mu, Sigma, r_f, sigma_P)
    )
