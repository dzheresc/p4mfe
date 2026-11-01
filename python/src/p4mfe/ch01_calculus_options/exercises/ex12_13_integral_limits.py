"""Exercises 12-13: limits of integrals.

12. (1/2h) int_{a-h}^{a+h} f -> f(a): mean value theorem for integrals; error
    O(h^2) for smooth f (central difference of an antiderivative).
13. int phi_{mu,sigma}(x) g(x) dx = int phi(z) g(mu + sigma z) dz -> g(mu) as
    sigma -> 0, by dominated convergence (|g| <= C).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..calculus import riemann_sum


def ex12_local_average(f: Callable, a: float, h: float, n: int = 64) -> float:
    return riemann_sum(f, a - h, a + h, n) / (2.0 * h)


def ex13_gaussian_average(g: Callable, mu: float, sigma: float, n: int = 4000, z_max: float = 10.0) -> float:
    def integrand(z):
        return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) * g(mu + sigma * z)

    return riemann_sum(integrand, -z_max, z_max, n)
