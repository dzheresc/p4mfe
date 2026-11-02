"""Sections 6.1-6.2.1: Taylor's formula for functions of one and several
variables.

    f(x) = sum_{k<=n} f^(k)(a) (x-a)^k / k! + R_n(x),
    R_n(x) = int_a^x (x-t)^n / n! f^(n+1)(t) dt,  |R_n| <= M |x-a|^(n+1) / (n+1)!

    f(x) ~ f(a) + grad f(a) . h + 1/2 h^T Hess f(a) h,   h = x - a.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from p4mfe.ch01_calculus_options.calculus import derivative, second_derivative
from p4mfe.ch01_calculus_options.multivariable import gradient, hessian, partial, second_partial
from p4mfe.ch02_integration_bonds.integration import integrate_to_tolerance


# 6.1 -----------------------------------------------------------------------
def taylor_polynomial(derivs: Sequence[float], a: float, x: float) -> float:
    """P_n(x) from f(a), f'(a), ..., f^(n)(a)."""
    if len(derivs) == 0:
        raise ValueError("need at least f(a)")
    h, term, total = x - a, 1.0, 0.0
    for k, d in enumerate(derivs):
        if k > 0:
            term *= h / k
        total += d * term
    return total


def taylor_linear(f: Callable[[float], float], a: float, x: float) -> float:
    return f(a) + derivative(f, a) * (x - a)


def taylor_quadratic(f: Callable[[float], float], a: float, x: float) -> float:
    h = x - a
    return f(a) + derivative(f, a) * h + 0.5 * second_derivative(f, a) * h * h


def taylor_remainder_bound(m: float, a: float, x: float, n: int) -> float:
    """M |x-a|^(n+1) / (n+1)!."""
    return m * abs(x - a) ** (n + 1) / math.factorial(n + 1)


def taylor_remainder_integral(
    dn1: Callable, a: float, x: float, n: int, tol: float = 1e-12
) -> float:
    """int_a^x (x-t)^n / n! f^(n+1)(t) dt by Simpson's rule (dn1 = f^(n+1), vectorized)."""
    if x == a:
        return 0.0
    nfact = math.factorial(n)
    lo, hi = min(a, x), max(a, x)
    v = integrate_to_tolerance(
        lambda t: (x - t) ** n / nfact * dn1(t), lo, hi, tol
    ).value
    return v if x > a else -v


# 6.2 -----------------------------------------------------------------------
def taylor_quadratic_from(fa: float, grad, hessian, h) -> float:
    """f(a) + g . h + 1/2 h^T H h for given gradient and Hessian."""
    g, hessian, h = (
        np.asarray(grad, float),
        np.asarray(hessian, float),
        np.asarray(h, float),
    )
    if g.shape != h.shape or hessian.shape != (h.size, h.size):
        raise ValueError("size mismatch")
    return float(fa + g @ h + 0.5 * h @ hessian @ h)


def taylor_quadratic_nd(f: Callable[[np.ndarray], float], a, x) -> float:
    a, x = np.asarray(a, float), np.asarray(x, float)
    return taylor_quadratic_from(f(a), gradient(f, a), hessian(f, a), x - a)


def taylor_linear_nd(f: Callable[[np.ndarray], float], a, x) -> float:
    a, x = np.asarray(a, float), np.asarray(x, float)
    return float(f(a) + gradient(f, a) @ (x - a))


# 6.2.1 ---------------------------------------------------------------------
@dataclass(frozen=True)
class Partials2D:
    f: float
    fx: float
    fy: float
    fxx: float
    fxy: float
    fyy: float


def partials_2d(
    f: Callable[[float, float], float], a: float, b: float
) -> Partials2D:
    g = lambda v: f(v[0], v[1])  # noqa: E731
    p = np.array([a, b], float)
    return Partials2D(
        f(a, b),
        partial(g, p, 0),
        partial(g, p, 1),
        second_partial(g, p, 0, 0),
        second_partial(g, p, 0, 1),
        second_partial(g, p, 1, 1),
    )


def taylor_linear_2d(d: Partials2D, dx: float, dy: float) -> float:
    return d.f + d.fx * dx + d.fy * dy


def taylor_quadratic_2d(d: Partials2D, dx: float, dy: float) -> float:
    return taylor_linear_2d(d, dx, dy) + 0.5 * (
        d.fxx * dx * dx + 2 * d.fxy * dx * dy + d.fyy * dy * dy
    )
