"""Exercises 5-9: moments of a binomial stock, uniform, exponential, normal |X|, Poisson.

5. S0 = 40, u = 1.05, d = 0.96, p = 0.6, three steps:
   E[S_T] = S0 (pu + (1-p)d)^3 = 41.703630,
   var(S_T) = S0^2 (pu^2 + (1-p)d^2)^3 - E[S_T]^2 = 9.883487.
6. Uniform on [a, b]: mean (a+b)/2, variance (b-a)^2/12.
7. Exponential: int f = 1, E[X] = 1/alpha, var = 1/alpha^2, F(x) = 1 - e^(-alpha x),
   P(X >= t) = e^(-alpha t), hence memoryless.
8. X ~ N(mu, sigma^2): E|X| = mu (1 - 2N(-mu/sigma)) + sigma sqrt(2/pi) e^(-mu^2/(2 sigma^2)),
   E[X^2] = mu^2 + sigma^2.
9. Poisson: E[X] = var(X) = lambda.

Full derivations: cpp/ch03_probability_black_scholes/exercises/include/p4mfe/ch03/exercises/ex05_09_distributions.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.probability import DiscreteDistribution
from p4mfe.normal import norm_cdf, norm_pdf


@dataclass(frozen=True)
class Ex05Tree:
    S0: float = 40.0
    u: float = 1.05
    d: float = 0.96
    p: float = 0.6
    steps: int = 3


def ex05_distribution(t: Ex05Tree = Ex05Tree()) -> DiscreteDistribution:
    n = t.steps
    values = [t.S0 * t.u**j * t.d ** (n - j) for j in range(n + 1)]
    probs = [
        math.comb(n, j) * t.p**j * (1 - t.p) ** (n - j) for j in range(n + 1)
    ]
    return DiscreteDistribution(values, probs)


def ex05_mean(t: Ex05Tree = Ex05Tree()) -> float:
    return t.S0 * (t.p * t.u + (1 - t.p) * t.d) ** t.steps


def ex05_variance(t: Ex05Tree = Ex05Tree()) -> float:
    m2 = t.S0**2 * (t.p * t.u**2 + (1 - t.p) * t.d**2) ** t.steps
    return m2 - ex05_mean(t) ** 2


def ex06_mean(a: float, b: float) -> float:
    return 0.5 * (a + b)


def ex06_variance(a: float, b: float) -> float:
    return (b - a) ** 2 / 12


def ex07_cdf(alpha: float, x: float) -> float:
    return 0.0 if x < 0 else -math.expm1(-alpha * x)


def ex07_tail(alpha: float, t: float) -> float:
    return 1.0 if t < 0 else math.exp(-alpha * t)


def ex07_conditional_tail(alpha: float, t: float, s: float) -> float:
    """P(X >= t + s | X >= t), equal to P(X >= s)."""
    return ex07_tail(alpha, t + s) / ex07_tail(alpha, t)


def ex08_mean_abs(mu: float, sigma: float) -> float:
    return mu * (1 - 2 * norm_cdf(-mu / sigma)) + 2 * sigma * norm_pdf(
        mu / sigma
    )


def ex08_second_moment(mu: float, sigma: float) -> float:
    return sigma**2 + mu**2


def ex09_distribution(lam: float) -> DiscreteDistribution:
    """Poisson probabilities, truncated once the tail is below 1e-15 and renormalized."""
    if lam <= 0:
        raise ValueError("lambda must be positive")
    values, probs = [], []
    pk, total, k = math.exp(-lam), 0.0, 0
    while True:
        values.append(float(k))
        probs.append(pk)
        total += pk
        if k > lam and 1 - total < 1e-15:
            break
        pk *= lam / (k + 1)
        k += 1
    return DiscreteDistribution(values, [p / total for p in probs])
