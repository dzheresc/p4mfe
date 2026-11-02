"""Sections 3.1-3.2.1: discrete and continuous random variables, expected
value, variance, covariance and correlation.

Continuous expectations E[g(X)] = int g(x) f(x) dx use the improper-integral
routine of chapter 2, so densities and g must accept numpy arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from p4mfe.ch02_integration_bonds.integration import improper_integral

ArrayFn = Callable[[np.ndarray], np.ndarray]


# 3.1 -----------------------------------------------------------------------
@dataclass(frozen=True)
class DiscreteDistribution:
    """P(X = values[i]) = probs[i]; probabilities nonnegative, summing to 1."""

    values: tuple[float, ...]
    probs: tuple[float, ...]

    def __init__(self, values: Sequence[float], probs: Sequence[float]):
        if len(values) != len(probs) or not values:
            raise ValueError("values and probs must match and be nonempty")
        if any(p < 0 for p in probs):
            raise ValueError("negative probability")
        if abs(math.fsum(probs) - 1.0) > 1e-12:
            raise ValueError("probabilities must sum to 1")
        object.__setattr__(self, "values", tuple(values))
        object.__setattr__(self, "probs", tuple(probs))

    def expected_value(self, g: Callable[[float], float]) -> float:
        """E[g(X)] = sum_i g(x_i) p_i."""
        return math.fsum(g(x) * p for x, p in zip(self.values, self.probs))

    def mean(self) -> float:
        return self.expected_value(lambda x: x)

    def variance(self) -> float:
        """E[(X - E[X])^2]."""
        m = self.mean()
        return self.expected_value(lambda x: (x - m) ** 2)


# 3.2 -----------------------------------------------------------------------
@dataclass(frozen=True)
class ContinuousDistribution:
    """Density pdf on [a, b]; either end may be infinite."""

    pdf: ArrayFn
    a: float = -math.inf
    b: float = math.inf

    def expected_value(self, g: ArrayFn, tol: float = 1e-9) -> float:
        """int_a^b g(x) f(x) dx; g is not evaluated where the density is 0,
        so fast-growing g cannot make 0 * inf = nan in the tails."""

        def integrand(x):
            p = self.pdf(x)
            with np.errstate(over="ignore", invalid="ignore"):
                return np.where(p == 0.0, 0.0, g(x) * p)

        return improper_integral(integrand, self.a, self.b, tol).value

    def mean(self, tol: float = 1e-9) -> float:
        return self.expected_value(lambda x: x, tol)

    def variance(self, tol: float = 1e-9) -> float:
        m = self.mean(tol)
        return self.expected_value(lambda x: (x - m) ** 2, tol)

    def cdf(self, t: float, tol: float = 1e-9) -> float:
        """F(t) = P(X <= t)."""
        if t <= self.a:
            return 0.0
        if t >= self.b:
            return 1.0
        return improper_integral(self.pdf, self.a, t, tol).value


def exponential_distribution(lam: float) -> ContinuousDistribution:
    """f(x) = lam e^(-lam x), x >= 0; mean 1/lam, variance 1/lam^2."""
    if lam <= 0:
        raise ValueError("lambda must be positive")
    return ContinuousDistribution(
        lambda x: np.where(x < 0, 0.0, lam * np.exp(-lam * np.maximum(x, 0.0))),
        0.0,
        math.inf,
    )


# 3.2.1 ---------------------------------------------------------------------
@dataclass(frozen=True)
class JointOutcome:
    """P(X = x, Y = y) = p."""

    x: float
    y: float
    p: float


def covariance(joint: Sequence[JointOutcome]) -> float:
    """cov(X, Y) = E[XY] - E[X] E[Y]."""
    ex = math.fsum(o.p * o.x for o in joint)
    ey = math.fsum(o.p * o.y for o in joint)
    return math.fsum(o.p * o.x * o.y for o in joint) - ex * ey


def correlation(joint: Sequence[JointOutcome]) -> float:
    """cov(X, Y) / (sigma_X sigma_Y), in [-1, 1]."""
    ex = math.fsum(o.p * o.x for o in joint)
    ey = math.fsum(o.p * o.y for o in joint)
    vx = math.fsum(o.p * (o.x - ex) ** 2 for o in joint)
    vy = math.fsum(o.p * (o.y - ey) ** 2 for o in joint)
    c = math.fsum(o.p * (o.x - ex) * (o.y - ey) for o in joint)
    return c / math.sqrt(vx * vy)


def variance_of_sum(
    a: float, var_x: float, b: float, var_y: float, cov_xy: float
) -> float:
    """var(aX + bY) = a^2 var(X) + b^2 var(Y) + 2ab cov(X, Y)."""
    return a * a * var_x + b * b * var_y + 2 * a * b * cov_xy


def sample_mean(x: Sequence[float]) -> float:
    if len(x) == 0:
        raise ValueError("empty sample")
    return float(np.mean(x))


def sample_covariance(x: Sequence[float], y: Sequence[float]) -> float:
    """Unbiased: divides by n - 1."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two samples of equal size >= 2")
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum((x - x.mean()) * (y - y.mean())) / (len(x) - 1))


def sample_variance(x: Sequence[float]) -> float:
    return sample_covariance(x, x)


def sample_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    return sample_covariance(x, y) / math.sqrt(
        sample_variance(x) * sample_variance(y)
    )
