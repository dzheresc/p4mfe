"""Section 8.4: relative extrema. A critical point is a minimum if the
Hessian is positive definite, a maximum if negative definite, a saddle if it
is nonsingular and neither; singular Hessians are inconclusive. Definiteness
by Sylvester's criterion on the leading principal minors. Critical points by
Newton's method (chapter 5) on grad f = 0.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

import numpy as np

from p4mfe.ch01_calculus_options.multivariable import gradient, hessian
from p4mfe.ch05_newton_bootstrapping.newton_nd import newton_nd
from p4mfe.ch05_newton_bootstrapping.solvers import Tolerances


class CriticalPointType(Enum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    SADDLE = "saddle"
    DEGENERATE = "degenerate"


def determinant(matrix) -> float:
    return float(np.linalg.det(np.asarray(matrix, float)))


def leading_principal_minors(hessian) -> list[float]:
    hessian = np.asarray(hessian, float)
    return [
        determinant(hessian[:k, :k]) for k in range(1, hessian.shape[0] + 1)
    ]


# 8.4 -----------------------------------------------------------------------
def classify_hessian(hessian, tol: float = 1e-6) -> CriticalPointType:
    minors = leading_principal_minors(hessian)
    if not minors:
        raise ValueError("empty matrix")
    if abs(minors[-1]) <= tol:
        return CriticalPointType.DEGENERATE
    if all(d > tol for d in minors):
        return CriticalPointType.MINIMUM
    if all(
        (d < -tol) if k % 2 == 0 else (d > tol) for k, d in enumerate(minors)
    ):
        return CriticalPointType.MAXIMUM
    return CriticalPointType.SADDLE


def classify_critical_point(
    f: Callable, x, tol: float = 1e-6
) -> CriticalPointType:
    return classify_hessian(hessian(f, x), tol)


def find_critical_point(
    f: Callable, x0, tol: Tolerances = Tolerances(1e-8, 1e-10, 100)
) -> np.ndarray:
    return newton_nd(
        lambda x: gradient(f, x), lambda x: hessian(f, x), x0, tol
    ).x
