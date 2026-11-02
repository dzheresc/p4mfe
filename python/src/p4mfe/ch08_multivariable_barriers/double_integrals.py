"""Sections 8.2, 8.3 and 8.6: double integrals, change of variables, polar
coordinates, and the integral of e^(-x^2/2).

Integrands f(x, y) must accept a scalar x and a numpy array y (most numpy
expressions do); both one-dimensional integrals use chapter 2's rules.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from p4mfe.ch02_integration_bonds.integration import Rule, integrate, simpson_rule

Fn2 = Callable[[float, np.ndarray], np.ndarray]


# 8.2 -----------------------------------------------------------------------
def double_integral(
    f: Fn2,
    a: float,
    b: float,
    c: float,
    d: float,
    nx: int = 200,
    ny: int = 200,
    rule: Rule = Rule.SIMPSON,
) -> float:
    """int_a^b int_c^d f(x, y) dy dx."""

    def inner(xs):
        return np.array(
            [
                integrate(lambda y: f(x, y), c, d, ny, rule)
                for x in np.atleast_1d(xs)
            ]
        )

    return integrate(inner, a, b, nx, rule)


def double_integral_region(
    f: Fn2,
    a: float,
    b: float,
    y_lo: Callable[[float], float],
    y_hi: Callable[[float], float],
    nx: int = 200,
    ny: int = 200,
    rule: Rule = Rule.SIMPSON,
) -> float:
    """int_a^b int_{y_lo(x)}^{y_hi(x)} f(x, y) dy dx."""

    def inner(xs):
        return np.array(
            [
                integrate(lambda y: f(x, y), y_lo(x), y_hi(x), ny, rule)
                for x in np.atleast_1d(xs)
            ]
        )

    return integrate(inner, a, b, nx, rule)


# 8.3 -----------------------------------------------------------------------
def jacobian_determinant(T: Callable, u, v):
    """|det J_T(u, v)| by central differences (u, v may be arrays)."""
    hu = 1e-6 * np.maximum(1.0, np.abs(u))
    hv = 1e-6 * np.maximum(1.0, np.abs(v))
    pu, mu = T(u + hu, v), T(u - hu, v)
    pv, mv = T(u, v + hv), T(u, v - hv)
    xu, yu = (pu[0] - mu[0]) / (2 * hu), (pu[1] - mu[1]) / (2 * hu)
    xv, yv = (pv[0] - mv[0]) / (2 * hv), (pv[1] - mv[1]) / (2 * hv)
    return np.abs(xu * yv - xv * yu)


def change_of_variables_integral(
    f: Fn2,
    T: Callable,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    nu: int = 200,
    nv: int = 200,
) -> float:
    def g(u, v):
        x, y = T(u, v)
        return f(x, y) * jacobian_determinant(T, u, v)

    return double_integral(g, u0, u1, v0, v1, nu, nv)


# 8.3.1 ---------------------------------------------------------------------
def from_polar(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def polar_integral(
    f: Fn2,
    r0: float,
    r1: float,
    theta0: float = 0.0,
    theta1: float = 2 * math.pi,
    nr: int = 200,
    ntheta: int = 200,
) -> float:
    """int int f(r cos theta, r sin theta) r dr dtheta."""

    def g(r, theta):
        x, y = from_polar(r, theta)
        return f(x, y) * r

    return double_integral(g, r0, r1, theta0, theta1, nr, ntheta)


# 8.6 -----------------------------------------------------------------------
def gaussian_disk_integral(R: float) -> float:
    """Integral of e^(-(x^2+y^2)/2) over the disk of radius R: 2 pi (1 - e^(-R^2/2))."""
    return -2 * math.pi * math.expm1(-0.5 * R * R)


def gaussian_square_integral(R: float, n: int = 400) -> float:
    """Integral over [-R, R]^2 = (int_{-R}^R e^(-x^2/2) dx)^2."""
    return simpson_rule(lambda x: np.exp(-0.5 * x * x), -R, R, n) ** 2
