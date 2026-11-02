"""Section 9.1: Lagrange multipliers.

Extrema of f(x) subject to g(x) = 0 (m < n constraints) solve
    grad f(x) + sum_j lambda_j grad g_j(x) = 0,  g(x) = 0,
here by Newton's method (chapter 5) with the bordered Jacobian
[[H_L, Jg^T], [Jg, 0]], H_L = H_f + sum_j lambda_j H_(g_j). The second-order
test classifies Z^T H_L Z, Z a basis of the tangent space {v : Jg v = 0},
with chapter 8's classify_hessian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from p4mfe.ch01_calculus_options.multivariable import gradient, hessian
from p4mfe.ch05_newton_bootstrapping.newton_nd import newton_nd
from p4mfe.ch05_newton_bootstrapping.solvers import Tolerances
from p4mfe.ch08_multivariable_barriers.extrema import CriticalPointType, classify_hessian

__all__ = [
    "CriticalPointType",
    "LagrangeResult",
    "classify_constrained",
    "lagrange_stationary_point",
    "lagrangian_gradient",
    "lagrangian_hessian",
    "least_squares_multipliers",
]


def _constraint_jacobian(g: Callable, x: np.ndarray) -> np.ndarray:
    m = len(g(x))
    return np.array([gradient(lambda y, j=j: g(y)[j], x) for j in range(m)])


@dataclass
class LagrangeResult:
    x: np.ndarray
    lam: np.ndarray
    iterations: int


# 9.1 -----------------------------------------------------------------------
def lagrangian_gradient(f: Callable, g: Callable, x, lam) -> np.ndarray:
    x = np.asarray(x, float)
    return np.concatenate(
        (
            gradient(f, x)
            + _constraint_jacobian(g, x).T @ np.asarray(lam, float),
            np.asarray(g(x), float),
        )
    )


def lagrangian_hessian(f: Callable, g: Callable, x, lam) -> np.ndarray:
    x = np.asarray(x, float)
    hess = hessian(f, x)
    for j, lj in enumerate(np.atleast_1d(lam)):
        hess = hess + lj * hessian(lambda y, j=j: g(y)[j], x)
    return hess


def least_squares_multipliers(f: Callable, g: Callable, x) -> np.ndarray:
    """(Jg Jg^T) lambda = -Jg grad f: the default starting multipliers."""
    x = np.asarray(x, float)
    jac = _constraint_jacobian(g, x)
    return np.linalg.solve(jac @ jac.T, -jac @ gradient(f, x))


def lagrange_stationary_point(
    f: Callable,
    g: Callable,
    x0,
    lam0=None,
    tol: Tolerances = Tolerances(1e-9, 1e-8, 100),
) -> LagrangeResult:
    x0 = np.asarray(x0, float)
    n, m = x0.size, len(g(x0))
    if not 0 < m < n:
        raise ValueError("need 0 < m < n constraints")
    lam0 = (
        least_squares_multipliers(f, g, x0)
        if lam0 is None
        else np.asarray(lam0, float)
    )

    def lagrange_system(z):
        return lagrangian_gradient(f, g, z[:n], z[n:])

    def jac(z):
        x, lam = z[:n], z[n:]
        jac_g = _constraint_jacobian(g, x)
        return np.block(
            [
                [lagrangian_hessian(f, g, x, lam), jac_g.T],
                [jac_g, np.zeros((m, m))],
            ]
        )

    res = newton_nd(lagrange_system, jac, np.concatenate((x0, lam0)), tol)
    return LagrangeResult(res.x[:n], res.x[n:], res.iterations)


def classify_constrained(
    f: Callable, g: Callable, x, lam, tol: float = 1e-6
) -> CriticalPointType:
    x = np.asarray(x, float)
    jac_g = _constraint_jacobian(g, x)
    # Orthonormal basis of the null space of Jg from the SVD.
    _, s, v_t = np.linalg.svd(jac_g)
    rank = int(np.sum(s > 1e-10))
    if rank < jac_g.shape[0]:
        raise ValueError("constraint gradients are linearly dependent")
    null_basis = v_t[rank:].T
    return classify_hessian(
        null_basis.T @ lagrangian_hessian(f, g, x, lam) @ null_basis, tol
    )
