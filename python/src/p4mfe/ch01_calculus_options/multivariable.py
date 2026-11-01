"""Section 1.6: functions of several variables (1.6.1 is the n = 2 case).

Partial derivatives, gradient and Hessian by central differences, for
f : R^n -> R taking a 1-D numpy array.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .calculus import _second_step, derivative, second_derivative

Func = Callable[[np.ndarray], float]


def partial(f: Func, x, i: int, h: float = 0.0) -> float:
    """df/dx_i at x."""
    x = np.asarray(x, dtype=float)
    y = x.copy()

    def along(xi: float) -> float:
        y[i] = xi
        return f(y)

    return derivative(along, x[i], h)


def second_partial(f: Func, x, i: int, j: int, h: float = 0.0) -> float:
    """d^2 f / dx_i dx_j at x (cross difference for i != j)."""
    x = np.asarray(x, dtype=float)
    y = x.copy()
    if i == j:

        def along(xi: float) -> float:
            y[i] = xi
            return f(y)

        return second_derivative(along, x[i], h)
    hi = h if h > 0 else _second_step(x[i])
    hj = h if h > 0 else _second_step(x[j])

    def at(si: int, sj: int) -> float:
        y[i] = x[i] + si * hi
        y[j] = x[j] + sj * hj
        return f(y)

    return (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1)) / (4 * hi * hj)


def gradient(f: Func, x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.array([partial(f, x, i) for i in range(x.size)])


def hessian(f: Func, x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            H[i, j] = H[j, i] = second_partial(f, x, i, j)
    return H
