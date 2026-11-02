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


def portfolio_variance(w, covariance) -> float:
    w = np.asarray(w, float)
    return float(w @ np.asarray(covariance, float) @ w)


def portfolio_std_dev(w, covariance) -> float:
    return math.sqrt(portfolio_variance(w, covariance))


@dataclass(frozen=True)
class FrontierConstants:
    a: float
    b: float
    c: float
    d: float


def frontier_constants(mu, covariance) -> FrontierConstants:
    mu = np.asarray(mu, float)
    ones = np.ones_like(mu)
    cov_inv_ones, cov_inv_mu = solve_linear(covariance, ones), solve_linear(
        covariance, mu
    )
    a, b, c = (
        float(ones @ cov_inv_ones),
        float(ones @ cov_inv_mu),
        float(mu @ cov_inv_mu),
    )
    d = a * c - b * b
    if not d > 0:
        raise ValueError("need non-constant expected returns")
    return FrontierConstants(a, b, c, d)


# 9.3 -----------------------------------------------------------------------
def global_min_variance_portfolio(covariance) -> np.ndarray:
    w = solve_linear(covariance, np.ones(len(covariance)))
    return w / w.sum()


def min_variance_portfolio(mu, covariance, target_mean: float) -> np.ndarray:
    c = frontier_constants(mu, covariance)
    rhs = (
        (c.c - c.b * target_mean)
        + (c.a * target_mean - c.b) * np.asarray(mu, float)
    ) / c.d
    return solve_linear(covariance, rhs)


def frontier_variance(mu, covariance, target_mean: float) -> float:
    c = frontier_constants(mu, covariance)
    return (c.a * target_mean**2 - 2 * c.b * target_mean + c.c) / c.d


def min_variance_portfolio_with_cash(
    mu, covariance, r_f: float, target_mean: float
) -> np.ndarray:
    """Risky weights; the cash weight is 1 - sum(w)."""
    m = np.asarray(mu, float) - r_f
    w = solve_linear(covariance, m)
    return w * (target_mean - r_f) / float(m @ w)


def tangency_portfolio(mu, covariance, r_f: float) -> np.ndarray:
    w = solve_linear(covariance, np.asarray(mu, float) - r_f)
    if w.sum() == 0:
        raise ValueError("undefined (r_f equals the GMV mean)")
    return w / w.sum()


def sharpe_ratio(w, mu, covariance, r_f: float) -> float:
    return (portfolio_mean(w, mu) - r_f) / portfolio_std_dev(w, covariance)


# 9.4 -----------------------------------------------------------------------
def max_return_for_std_dev(mu, covariance, sigma_p: float) -> float:
    c = frontier_constants(mu, covariance)
    disc = c.a * sigma_p**2 - 1
    if disc < 0:
        raise ValueError("sigma_P below the minimum 1/sqrt(A)")
    return (c.b + math.sqrt(c.d * disc)) / c.a


def max_return_portfolio(mu, covariance, sigma_p: float) -> np.ndarray:
    return min_variance_portfolio(
        mu, covariance, max_return_for_std_dev(mu, covariance, sigma_p)
    )


def max_return_with_cash(mu, covariance, r_f: float, sigma_p: float) -> float:
    m = np.asarray(mu, float) - r_f
    return r_f + sigma_p * math.sqrt(float(m @ solve_linear(covariance, m)))


def max_return_portfolio_with_cash(
    mu, covariance, r_f: float, sigma_p: float
) -> np.ndarray:
    return min_variance_portfolio_with_cash(
        mu, covariance, r_f, max_return_with_cash(mu, covariance, r_f, sigma_p)
    )
