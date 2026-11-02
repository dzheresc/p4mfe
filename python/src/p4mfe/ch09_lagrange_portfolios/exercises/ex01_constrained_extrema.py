"""Exercise 1: extrema of f = x1 - 2x2 + 3x3 - 4x4 subject to
x1^2 + x2^2 = x4^2 + 1 and x1^2 + 2x2^2 + 3x3^2 = 6.

The feasible set is compact and the constraint gradients are independent on
it, so the extrema solve the Lagrange system; x1 = -1/(2(l1 + l2)),
x2 = 1/(l1 + 2 l2), x3 = -1/(2 l2), x4 = -2/l1. Newton's method from the grid
{-2, -1, 1, 2}^4 finds four stationary points, f = +-12.761255 (max/min) and
+-3.434511 (saddles); f is odd and g even, so min f = -max f.

Full derivation: cpp/ch09_lagrange_portfolios/exercises/include/p4mfe/ch09/exercises/ex01_constrained_extrema.hpp
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from p4mfe.ch09_lagrange_portfolios.lagrange import CriticalPointType, classify_constrained, lagrange_stationary_point


def ex01_f(x) -> float:
    return float(x[0] - 2 * x[1] + 3 * x[2] - 4 * x[3])


def ex01_g(x) -> np.ndarray:
    return np.array(
        [
            x[0] ** 2 + x[1] ** 2 - x[3] ** 2 - 1,
            x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2 - 6,
        ]
    )


def ex01_x_from_lambda(lam) -> np.ndarray:
    l1, l2 = lam
    return np.array(
        [-1 / (2 * (l1 + l2)), 1 / (l1 + 2 * l2), -1 / (2 * l2), -2 / l1]
    )


@dataclass
class Ex01Point:
    x: np.ndarray
    lam: np.ndarray
    value: float
    type: CriticalPointType


def ex01_stationary_points() -> list[Ex01Point]:
    """Stationary points from Newton's method on the grid {-2,-1,1,2}^4, sorted by decreasing f."""
    pts: list[Ex01Point] = []
    for x0 in itertools.product((-2.0, -1.0, 1.0, 2.0), repeat=4):
        try:
            r = lagrange_stationary_point(ex01_f, ex01_g, x0)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        if np.max(np.abs(ex01_g(r.x))) > 1e-8:
            continue
        if any(np.sum((p.x - r.x) ** 2) < 1e-10 for p in pts):
            continue
        pts.append(
            Ex01Point(
                r.x,
                r.lam,
                ex01_f(r.x),
                classify_constrained(ex01_f, ex01_g, r.x, r.lam),
            )
        )
    return sorted(pts, key=lambda p: -p.value)


def ex01_sweep(n_t: int, n_p: int) -> tuple[float, float]:
    """(max f, min f) over a grid of the ellipsoid, x4 = -/+ sqrt(x1^2 + x2^2 - 1)."""
    t = np.linspace(0, math.pi, n_t + 1)[:, None]
    p = 2 * math.pi * np.arange(n_p)[None, :] / n_p
    x1 = math.sqrt(6) * np.sin(t) * np.cos(p)
    x2 = math.sqrt(3) * np.sin(t) * np.sin(p)
    x3 = math.sqrt(2) * np.cos(t) * np.ones_like(p)
    r = x1**2 + x2**2 - 1
    ok = r >= 0
    lin, q = (x1 - 2 * x2 + 3 * x3)[ok], 4 * np.sqrt(r[ok])
    return float(np.max(lin + q)), float(np.min(lin - q))
