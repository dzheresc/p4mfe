"""Section 7.2: finite difference solutions of ODEs.

Initial value problem y' = f(x, y), y(x0) = y0, on n equal steps:
    forward  (explicit Euler)  y_{k+1} = y_k + h f(x_k, y_k)              O(h)
    backward (implicit Euler)  y_{k+1} = y_k + h f(x_{k+1}, y_{k+1})      O(h)
             (each step solved by chapter 5's Newton's method)
    central  (leapfrog)        y_{k+1} = y_{k-1} + 2h f(x_k, y_k)         O(h^2)

Boundary value problem y'' + p(x) y' + q(x) y = r(x), y(a) = ya, y(b) = yb:
central differences at the interior nodes give a tridiagonal system, solved
by the Thomas algorithm; O(h^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from p4mfe.ch01_calculus_options.calculus import derivative
from p4mfe.ch05_newton_bootstrapping.solvers import Tolerances, newton

OdeFn = Callable[[float, float], float]


@dataclass
class OdeSolution:
    x: np.ndarray
    y: np.ndarray


class OdeScheme(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    CENTRAL = "central"


def _grid(x0: float, x_end: float, n: int) -> OdeSolution:
    if n < 1:
        raise ValueError("need n >= 1 steps")
    x = np.linspace(x0, x_end, n + 1)
    return OdeSolution(x, np.zeros(n + 1))


# 7.2 -----------------------------------------------------------------------
def euler_forward(f: OdeFn, x0: float, y0: float, x_end: float, n: int) -> OdeSolution:
    s = _grid(x0, x_end, n)
    h = (x_end - x0) / n
    s.y[0] = y0
    for k in range(n):
        s.y[k + 1] = s.y[k] + h * f(s.x[k], s.y[k])
    return s


def euler_backward(f: OdeFn, x0: float, y0: float, x_end: float, n: int) -> OdeSolution:
    """Solves y - y_k - h f(x_{k+1}, y) = 0 by Newton from the explicit guess."""
    s = _grid(x0, x_end, n)
    h = (x_end - x0) / n
    s.y[0] = y0
    for k in range(n):
        yk, x1 = float(s.y[k]), float(s.x[k + 1])

        def g(y, yk=yk, x1=x1):
            return y - yk - h * f(x1, y)

        scale = max(1.0, abs(yk))
        tol = Tolerances(1e-12 * scale, 1e-12 * scale, 100)
        s.y[k + 1] = newton(g, lambda y, g=g: derivative(g, y), yk + h * f(float(s.x[k]), yk), tol).x
    return s


def leapfrog(f: OdeFn, x0: float, y0: float, x_end: float, n: int) -> OdeSolution:
    s = _grid(x0, x_end, n)
    h = (x_end - x0) / n
    s.y[0] = y0
    s.y[1] = y0 + h * f(x0, y0)
    for k in range(1, n):
        s.y[k + 1] = s.y[k - 1] + 2 * h * f(s.x[k], s.y[k])
    return s


def solve_ivp(f: OdeFn, x0: float, y0: float, x_end: float, n: int, scheme: OdeScheme) -> OdeSolution:
    if scheme is OdeScheme.FORWARD:
        return euler_forward(f, x0, y0, x_end, n)
    if scheme is OdeScheme.BACKWARD:
        return euler_backward(f, x0, y0, x_end, n)
    return leapfrog(f, x0, y0, x_end, n)


def solve_tridiagonal(lower, diag, upper, rhs) -> np.ndarray:
    """Thomas algorithm: lower[i] x[i-1] + diag[i] x[i] + upper[i] x[i+1] = rhs[i]."""
    d = np.array(diag, float)
    b = np.array(rhs, float)
    lo = np.asarray(lower, float)
    up = np.asarray(upper, float)
    n = d.size
    if n == 0 or lo.size != n or up.size != n or b.size != n:
        raise ValueError("size mismatch")
    for i in range(1, n):
        if d[i - 1] == 0:
            raise ZeroDivisionError("zero pivot")
        m = lo[i] / d[i - 1]
        d[i] -= m * up[i - 1]
        b[i] -= m * b[i - 1]
    x = np.empty(n)
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - up[i] * x[i + 1]) / d[i]
    return x


def solve_bvp(p: Callable[[float], float], q: Callable[[float], float], r: Callable[[float], float],
              a: float, b: float, ya: float, yb: float, n: int) -> OdeSolution:
    """y'' + p y' + q y = r on [a, b], y(a) = ya, y(b) = yb, n intervals."""
    if n < 2:
        raise ValueError("need n >= 2 intervals")
    s = _grid(a, b, n)
    h = (b - a) / n
    xi = s.x[1:-1]
    pi = np.array([p(x) for x in xi])
    lo = 1 / h**2 - pi / (2 * h)
    up = 1 / h**2 + pi / (2 * h)
    di = np.array([q(x) for x in xi]) - 2 / h**2
    rhs = np.array([r(x) for x in xi])
    rhs[0] -= lo[0] * ya
    rhs[-1] -= up[-1] * yb
    s.y[0], s.y[-1] = ya, yb
    s.y[1:-1] = solve_tridiagonal(lo, di, up, rhs)
    return s


def max_error(s: OdeSolution, exact: Callable[[float], float]) -> float:
    return float(max(abs(y - exact(x)) for x, y in zip(s.x, s.y)))
