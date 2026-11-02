"""Section 10.2: polynomial interpolation.

The unique polynomial of degree <= n through n+1 distinct nodes, in
    Lagrange form  p(x) = sum y_i L_i(x), L_i(x) = prod_{j!=i} (x-x_j)/(x_i-x_j),
    Newton form    p(x) = sum c_k prod_{j<k} (x - x_j), c_k = f[x_0..x_k],
    monomial form  from the Vandermonde system (chapter 5's solve_linear).
Error: |f(x) - p(x)| <= M / (n+1)! |prod (x - x_i)| when |f^(n+1)| <= M.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from p4mfe.ch05_newton_bootstrapping.newton_nd import solve_linear


def _check_nodes(xs: Sequence[float], ny: int) -> None:
    if len(xs) == 0 or len(xs) != ny:
        raise ValueError("need as many values as nodes")
    if len(set(xs)) != len(xs):
        raise ValueError("nodes must be distinct")


def polynomial_value(a: Sequence[float], x: float) -> float:
    """sum a_k x^k by Horner's rule (a_0 first)."""
    v = 0.0
    for c in reversed(a):
        v = v * x + c
    return v


def lagrange_basis(xs: Sequence[float], i: int, x: float) -> float:
    v = 1.0
    for j, xj in enumerate(xs):
        if j != i:
            v *= (x - xj) / (xs[i] - xj)
    return v


def lagrange_interpolate(
    xs: Sequence[float], ys: Sequence[float], x: float
) -> float:
    _check_nodes(xs, len(ys))
    return sum(y * lagrange_basis(xs, i, x) for i, y in enumerate(ys))


def divided_differences(
    xs: Sequence[float], ys: Sequence[float]
) -> list[float]:
    """Newton coefficients c_k = f[x_0, ..., x_k]."""
    _check_nodes(xs, len(ys))
    c = [float(y) for y in ys]
    n = len(xs)
    for k in range(1, n):
        for i in range(n - 1, k - 1, -1):
            c[i] = (c[i] - c[i - 1]) / (xs[i] - xs[i - k])
    return c


def newton_interpolate(
    xs: Sequence[float], c: Sequence[float], x: float
) -> float:
    if len(c) == 0 or len(c) > len(xs):
        raise ValueError("size mismatch")
    v = c[-1]
    for k in range(len(c) - 2, -1, -1):
        v = v * (x - xs[k]) + c[k]
    return v


def interpolating_polynomial(
    xs: Sequence[float], ys: Sequence[float]
) -> np.ndarray:
    """Monomial coefficients a_0..a_n (Vandermonde system; small n only)."""
    _check_nodes(xs, len(ys))
    vandermonde = np.vander(np.asarray(xs, float), increasing=True)
    return solve_linear(vandermonde, ys)


def interpolation_error_bound(
    deriv_bound: float, xs: Sequence[float], x: float
) -> float:
    return (
        deriv_bound
        * abs(math.prod(x - xi for xi in xs))
        / math.factorial(len(xs))
    )
