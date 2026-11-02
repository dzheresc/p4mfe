"""Exercise 5: approximate Newton with central difference Jacobians (h = 1e-6) vs forward differences.

From (1, 2, 3): (-1.690551, 1.983107, -0.884558) in 9 iterations with either scheme.
From (2, 2, 2): a long chaotic excursion, usually ending at the exact root (-1, 3, 1) (here 127 central,
81 forward, 40 with the exact Jacobian); the step count and even the root depend on the scheme and on rounding.

Full derivation: cpp/ch05_newton_bootstrapping/exercises/include/p4mfe/ch05/exercises/ex05_central_difference_newton.hpp
"""

from __future__ import annotations

import numpy as np

from ..newton_nd import Difference, NdRootResult, approximate_newton_nd, newton_nd
from ..solvers import Tolerances


def ex05_f(x) -> np.ndarray:
    x1, x2, x3 = x
    return np.array(
        [
            x1**3 + 2 * x1 * x2 + x3**2 - x2 * x3 + 9,
            2 * x1**2 + 2 * x1 * x2**2 + x2**3 * x3**2 - x2**2 * x3 - 2,
            x1 * x2 * x3 + x1**3 - x3**2 - x1 * x2**2 - 4,
        ]
    )


def ex05_jacobian(x) -> np.ndarray:
    x1, x2, x3 = x
    return np.array(
        [
            [3 * x1**2 + 2 * x2, 2 * x1 - x3, 2 * x3 - x2],
            [
                4 * x1 + 2 * x2**2,
                4 * x1 * x2 + 3 * x2**2 * x3**2 - 2 * x2 * x3,
                2 * x2**3 * x3 - x2**2,
            ],
            [
                x2 * x3 + 3 * x1**2 - x2**2,
                x1 * x3 - 2 * x1 * x2,
                x1 * x2 - 2 * x3,
            ],
        ]
    )


EX05_TOLERANCES = Tolerances(1e-9, 1e-6, 1000)
EX05_H = 1e-6


def ex05_solve(x0, scheme: Difference = Difference.CENTRAL) -> NdRootResult:
    return approximate_newton_nd(
        ex05_f, np.asarray(x0, dtype=float), EX05_TOLERANCES, EX05_H, scheme
    )


def ex05_solve_exact(x0) -> NdRootResult:
    return newton_nd(
        ex05_f, ex05_jacobian, np.asarray(x0, dtype=float), EX05_TOLERANCES
    )
