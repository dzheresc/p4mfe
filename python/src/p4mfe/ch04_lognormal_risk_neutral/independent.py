"""Sections 4.3-4.3.2: independent random variables.

Independence means f(x, y) = f_X(x) f_Y(y); the density of X + Y is the
convolution int f_X(x) f_Y(z - x) dx. Linear combinations of independent
normals are normal (4.3.1), so products of powers of independent lognormals
are lognormal (4.3.2).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from p4mfe.ch02_integration_bonds.integration import improper_integral
from p4mfe.ch03_probability_black_scholes.normal_rv import NormalParams

ArrayFn = Callable[[np.ndarray], np.ndarray]


# 4.3 -----------------------------------------------------------------------
def independent_joint_pdf(f_x: ArrayFn, f_y: ArrayFn) -> Callable:
    """(x, y) -> f_X(x) f_Y(y)."""
    return lambda x, y: f_x(x) * f_y(y)


def sum_pdf(
    f_x: ArrayFn,
    f_y: ArrayFn,
    z: float,
    lo: float = -math.inf,
    hi: float = math.inf,
    tol: float = 1e-10,
) -> float:
    """f_{X+Y}(z) = int_lo^hi f_X(x) f_Y(z - x) dx; [lo, hi] is where the
    integrand can be nonzero (pass it for densities with bounded support)."""
    if not lo < hi:
        return 0.0
    return improper_integral(lambda x: f_x(x) * f_y(z - x), lo, hi, tol).value


# 4.3.1 ---------------------------------------------------------------------
def independent_normal_combination(
    a: Sequence[float], x_dist: Sequence[NormalParams]
) -> NormalParams:
    """sum a_i X_i: mean sum a_i mu_i, variance sum a_i^2 sigma_i^2."""
    if len(a) != len(x_dist):
        raise ValueError("sizes differ")
    return NormalParams(
        math.fsum(ai * x.mean for ai, x in zip(a, x_dist)),
        math.fsum(ai * ai * x.variance for ai, x in zip(a, x_dist)),
    )


def independent_normal_sum(x_dist: Sequence[NormalParams]) -> NormalParams:
    return independent_normal_combination([1.0] * len(x_dist), x_dist)


# 4.3.2 ---------------------------------------------------------------------
def independent_lognormal_product(
    log_y: Sequence[NormalParams], a: Sequence[float] | None = None
) -> NormalParams:
    """prod Y_i^(a_i) (a_i = 1 by default) for independent lognormal Y_i given
    by the normal parameters of ln(Y_i); returns those of ln(product)."""
    return independent_normal_combination(
        [1.0] * len(log_y) if a is None else a, log_y
    )


__all__ = [
    "NormalParams",
    "independent_joint_pdf",
    "independent_lognormal_product",
    "independent_normal_combination",
    "independent_normal_sum",
    "sum_pdf",
]
