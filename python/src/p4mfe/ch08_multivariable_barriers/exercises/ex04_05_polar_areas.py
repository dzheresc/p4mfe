"""Exercises 4-5: in polar coordinates det J = r, so the disk of radius R has
area int_0^{2 pi} int_0^R r dr dtheta = pi R^2; with x = a r cos theta,
y = b r sin theta, det J = a b r and the ellipse has area pi a b.

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex04_05_polar_areas.hpp
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch08_multivariable_barriers.double_integrals import (
    change_of_variables_integral,
    double_integral_region,
    polar_integral,
)


def _one(x, y):
    return np.ones_like(np.asarray(y, float))


def ex04_disk_area_polar(R: float, n: int = 50) -> float:
    return polar_integral(_one, 0.0, R, 0.0, 2 * math.pi, n, n)


def ex04_disk_area_cartesian(R: float, n: int = 2000) -> float:
    return double_integral_region(
        _one,
        -R,
        R,
        lambda x: -math.sqrt(max(R * R - x * x, 0.0)),
        lambda x: math.sqrt(max(R * R - x * x, 0.0)),
        n,
        2,
    )


def ex05_map(a: float, b: float, r, theta):
    return a * r * np.cos(theta), b * r * np.sin(theta)


def ex05_ellipse_area(a: float, b: float, n: int = 50) -> float:
    return change_of_variables_integral(
        _one,
        lambda r, th: ex05_map(a, b, r, th),
        0.0,
        1.0,
        0.0,
        2 * math.pi,
        n,
        n,
    )


def ex05_exact(a: float, b: float) -> float:
    return math.pi * a * b
