"""Section 8.5: Theta of a derivative security.

Theta = dV/dt = -dV/dtau with tau = T - t (chain rule). For payoff h,
V(tau) = e^(-r tau) int h(S e^((r - q - sigma^2/2) tau + sigma sqrt(tau) z)) phi(z) dz,
differentiated under the integral sign (chapter 2).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from p4mfe.ch01_calculus_options.calculus import derivative
from p4mfe.ch02_integration_bonds.integration import improper_integral_derivative
from p4mfe.ch04_lognormal_risk_neutral.asset_model import price_from_normal

_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)


# 8.5 -----------------------------------------------------------------------
def theta_from_maturity(V: Callable[[float], float], tau: float) -> float:
    return -derivative(V, tau)


def theta_by_integral(h: Callable, S: float, T: float, sigma: float, r: float, q: float = 0.0,
                      tol: float = 1e-7) -> float:
    """h must accept numpy arrays."""
    def integrand(z, tau):
        p = np.exp(-0.5 * z * z) * _INV_SQRT_2PI
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(p == 0.0, 0.0, math.exp(-r * tau) * h(price_from_normal(z, S, tau, sigma, r, q)) * p)

    return -improper_integral_derivative(integrand, -math.inf, math.inf, T, tol)
