"""Exercise 3: five assets, long and short positions allowed.

(i)  minimum variance for mean 15%: w = (0.067354, 0.217299, -0.002244, 0.406893, 0.310697),
     sigma = 14.12%;
(ii) maximum mean for sigma 25%: mu = 19.92%,
     w = (-0.395503, 0.096062, 0.045839, 0.709880, 0.543722).

Full derivation: cpp/ch09_lagrange_portfolios/exercises/include/p4mfe/ch09/exercises/ex03_five_asset_portfolio.hpp
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from p4mfe.ch09_lagrange_portfolios.portfolio import (
    covariance_matrix,
    max_return_portfolio,
    min_variance_portfolio,
    portfolio_mean,
    portfolio_std_dev,
)


@dataclass(frozen=True)
class Ex03Data:
    mu: np.ndarray = field(
        default_factory=lambda: np.array([0.08, 0.10, 0.13, 0.15, 0.20])
    )
    sigma: np.ndarray = field(
        default_factory=lambda: np.array([0.14, 0.18, 0.23, 0.25, 0.35])
    )
    corr: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.0, -0.3, 0.4, 0.25, -0.2],
                [-0.3, 1.0, -0.1, -0.2, 0.15],
                [0.4, -0.1, 1.0, 0.35, 0.25],
                [0.25, -0.2, 0.35, 1.0, -0.15],
                [-0.2, 0.15, 0.25, -0.15, 1.0],
            ]
        )
    )

    def covariance(self) -> np.ndarray:
        return covariance_matrix(self.sigma, self.corr)


@dataclass
class Ex03Portfolio:
    w: np.ndarray
    mean: float
    std_dev: float


def ex03_min_variance(
    target_mean: float = 0.15, d: Ex03Data = Ex03Data()
) -> Ex03Portfolio:
    S = d.covariance()
    w = min_variance_portfolio(d.mu, S, target_mean)
    return Ex03Portfolio(w, portfolio_mean(w, d.mu), portfolio_std_dev(w, S))


def ex03_max_return(
    target_std: float = 0.25, d: Ex03Data = Ex03Data()
) -> Ex03Portfolio:
    S = d.covariance()
    w = max_return_portfolio(d.mu, S, target_std)
    return Ex03Portfolio(w, portfolio_mean(w, d.mu), portfolio_std_dev(w, S))
