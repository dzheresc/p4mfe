"""Exercise 9: the Taylor series of ln(1 + x) converges to ln(1 + x) for every |x| < 1.

f(x) - P_n(x) = int_0^x (-1)^n (x-t)^n/(1+t)^(n+1) dt, and |f(x) - P_n(x)| <= |x|^n |ln(1 + x)| -> 0.

Full proof: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex09_log_series_convergence.hpp
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch02_integration_bonds.integration import integrate_to_tolerance

from ..taylor_series import log1p_series


def ex09_error(x: float, n: int) -> float:
    return math.log1p(x) - log1p_series(x, n)


def ex09_error_integral(x: float, n: int, tol: float = 1e-13) -> float:
    if not x > -1:
        raise ValueError("need x > -1")
    if x == 0:
        return 0.0
    sign = 1.0 if n % 2 == 0 else -1.0
    v = integrate_to_tolerance(
        lambda t: sign * (x - t) ** n / (1.0 + t) ** (n + 1),
        min(0.0, x),
        max(0.0, x),
        tol,
    ).value
    return v if x > 0 else -v


def ex09_middle_bound(x: float, n: int, tol: float = 1e-13) -> float:
    if not 0 <= x < 1:
        raise ValueError("need 0 <= x < 1")
    if x == 0:
        return 0.0
    return integrate_to_tolerance(
        lambda t: ((x - t) / (1.0 + t)) ** n / (1.0 + t), 0.0, x, tol
    ).value


def ex09_bound(x: float, n: int) -> float:
    if not abs(x) < 1:
        raise ValueError("need |x| < 1")
    return abs(x) ** n * abs(math.log1p(x))


def ex09_error_negative(x: float, n: int, tol: float = 1e-13) -> float:
    if not -1 < x <= 0:
        raise ValueError("need -1 < x <= 0")
    s = -x
    if s == 0:
        return 0.0
    return integrate_to_tolerance(
        lambda z: np.power(s - z, n) / np.power(1.0 - z, n + 1), 0.0, s, tol
    ).value
