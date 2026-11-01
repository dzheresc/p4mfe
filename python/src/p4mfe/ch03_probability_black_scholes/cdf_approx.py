"""Section 3.9: computing N(x) to implement the Black-Scholes formula.

- ``norm_cdf_poly``: Abramowitz & Stegun 26.2.17, absolute error < 7.5e-8;
- ``norm_cdf_simpson``: N(x) = 1/2 + 1/sqrt(2 pi) int_0^x e^(-t^2/2) dt with
  Simpson's rule, n doubled until the tolerance is met (section 2.4.1).
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch02_integration_bonds.integration import integrate_to_tolerance
from p4mfe.normal import norm_pdf

_A = (0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429)


def norm_cdf_poly(x: float) -> float:
    """1 - phi(x) (a1 z + ... + a5 z^5), z = 1/(1 + 0.2316419 x), x >= 0."""
    if x < 0:
        return 1.0 - norm_cdf_poly(-x)
    z = 1.0 / (1.0 + 0.2316419 * x)
    poly = 0.0
    for a in reversed(_A):  # Horner
        poly = (poly + a) * z
    return 1.0 - norm_pdf(x) * poly


def norm_cdf_simpson(x: float, tol: float = 1e-12) -> float:
    if x == 0:
        return 0.5
    integral = integrate_to_tolerance(lambda t: np.exp(-0.5 * t * t), 0.0, abs(x), tol).value
    half_mass = integral / math.sqrt(2 * math.pi)
    return 0.5 + half_mass if x > 0 else 0.5 - half_mass
