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
    return np.concatenate((gradient(f, x) + _constraint_jacobian(g, x).T @ np.asarray(lam, float),
                           np.asarray(g(x), float)))


def lagrangian_hessian(f: Callable, g: Callable, x, lam) -> np.ndarray:
    x = np.asarray(x, float)
    H = hessian(f, x)
    for j, lj in enumerate(np.atleast_1d(lam)):
        H = H + lj * hessian(lambda y, j=j: g(y)[j], x)
    return H


def least_squares_multipliers(f: Callable, g: Callable, x) -> np.ndarray:
    """(Jg Jg^T) lambda = -Jg grad f: the default starting multipliers."""
    x = np.asarray(x, float)
    J = _constraint_jacobian(g, x)
    return np.linalg.solve(J @ J.T, -J @ gradient(f, x))


def lagrange_stationary_point(f: Callable, g: Callable, x0, lam0=None,
                              tol: Tolerances = Tolerances(1e-9, 1e-8, 100)) -> LagrangeResult:
    x0 = np.asarray(x0, float)
    n, m = x0.size, len(g(x0))
    if not 0 < m < n:
        raise ValueError("need 0 < m < n constraints")
    lam0 = least_squares_multipliers(f, g, x0) if lam0 is None else np.asarray(lam0, float)

    def F(z):
        return lagrangian_gradient(f, g, z[:n], z[n:])

    def J(z):
        x, lam = z[:n], z[n:]
        Jg = _constraint_jacobian(g, x)
        return np.block([[lagrangian_hessian(f, g, x, lam), Jg.T], [Jg, np.zeros((m, m))]])

    res = newton_nd(F, J, np.concatenate((x0, lam0)), tol)
    return LagrangeResult(res.x[:n], res.x[n:], res.iterations)


def classify_constrained(f: Callable, g: Callable, x, lam, tol: float = 1e-6) -> CriticalPointType:
    x = np.asarray(x, float)
    Jg = _constraint_jacobian(g, x)
    # Orthonormal basis of the null space of Jg from the SVD.
    _, s, Vt = np.linalg.svd(Jg)
    rank = int(np.sum(s > 1e-10))
    if rank < Jg.shape[0]:
        raise ValueError("constraint gradients are linearly dependent")
    Z = Vt[rank:].T
    return classify_hessian(Z.T @ lagrangian_hessian(f, g, x, lam) @ Z, tol)
