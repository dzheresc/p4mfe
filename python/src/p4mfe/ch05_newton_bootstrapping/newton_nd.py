"""Section 5.2: Newton's method for F(x) = 0, F: R^N -> R^N.

- newton_nd (5.2.1):             x <- x - D_F(x)^{-1} F(x), solving the linear
                                 system D_F(x) s = -F(x) rather than inverting
- approximate_newton_nd (5.2.2): D_F replaced by finite differences
                                 (F_i(x + h e_j) - F_i(x)) / h

Stops when ||F(x_k)|| <= tol.approx and ||x_k - x_{k-1}|| <= tol.consec.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .solvers import Tolerances

VectorFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class NdRootResult:
    x: np.ndarray
    iterations: int


def solve_linear(a, b) -> np.ndarray:
    """A x = b (LU with partial pivoting, via numpy)."""
    a = np.asarray(a, dtype=float)
    try:
        return np.linalg.solve(a, np.asarray(b, dtype=float))
    except np.linalg.LinAlgError as e:
        raise RuntimeError("singular matrix") from e


def newton_nd(
    func: VectorFn, jacobian: Callable, x0, tol: Tolerances = Tolerances()
) -> NdRootResult:
    x = np.array(x0, dtype=float)
    fx = np.asarray(func(x), dtype=float)
    if fx.shape != x.shape:
        raise ValueError("F must map R^N to R^N")
    k = 0
    while k < tol.max_iter:
        s = solve_linear(jacobian(x), -fx)
        x = x + s
        k += 1
        fx = np.asarray(func(x), dtype=float)
        step = float(np.linalg.norm(s))
        if (
            np.linalg.norm(fx) <= tol.approx and step <= tol.consec
        ) or step == 0:
            return NdRootResult(x, k)
    raise RuntimeError("newton_nd: no convergence within max_iter")


class Difference(Enum):
    FORWARD = "forward"
    CENTRAL = "central"


def finite_difference_jacobian(
    func: VectorFn, x, h: float = 1e-6, scheme: Difference = Difference.FORWARD
) -> np.ndarray:
    """Forward differences are O(h) accurate (the book's choice); central ones O(h^2)."""
    x = np.array(x, dtype=float)
    f0 = np.asarray(func(x), dtype=float)
    jacobian = np.empty((f0.size, x.size))
    for j in range(x.size):
        e = np.zeros_like(x)
        e[j] = h
        if scheme is Difference.FORWARD:
            jacobian[:, j] = (np.asarray(func(x + e)) - f0) / h
        else:
            jacobian[:, j] = (
                np.asarray(func(x + e)) - np.asarray(func(x - e))
            ) / (2 * h)
    return jacobian


def approximate_newton_nd(
    func: VectorFn,
    x0,
    tol: Tolerances = Tolerances(),
    h: float = 1e-6,
    scheme: Difference = Difference.FORWARD,
) -> NdRootResult:
    return newton_nd(
        func, lambda x: finite_difference_jacobian(func, x, h, scheme), x0, tol
    )
