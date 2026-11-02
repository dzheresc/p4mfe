"""Section 8.1: the chain rule for functions of several variables.

d/dt f(g(t)) = grad f(g(t)) . g'(t)
grad (f o G)(x) = J_G(x)^T grad f(G(x))
J_(F o G)(x) = J_F(G(x)) J_G(x)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from p4mfe.ch01_calculus_options.calculus import derivative
from p4mfe.ch01_calculus_options.multivariable import gradient
from p4mfe.ch05_newton_bootstrapping.newton_nd import Difference, finite_difference_jacobian


# 8.1 -----------------------------------------------------------------------
def curve_derivative(g: Callable[[float], np.ndarray], t: float) -> np.ndarray:
    n = len(g(t))
    return np.array([derivative(lambda s, i=i: g(s)[i], t) for i in range(n)])


def chain_rule_derivative(
    f: Callable, g: Callable[[float], np.ndarray], t: float
) -> float:
    return float(np.dot(gradient(f, g(t)), curve_derivative(g, t)))


def chain_rule_gradient(f: Callable, G: Callable, x) -> np.ndarray:
    J = finite_difference_jacobian(G, x, 1e-6, Difference.CENTRAL)
    return J.T @ gradient(f, np.asarray(G(np.asarray(x, float))))


def chain_rule_jacobian(F: Callable, G: Callable, x) -> np.ndarray:
    x = np.asarray(x, float)
    return finite_difference_jacobian(
        F, np.asarray(G(x)), 1e-6, Difference.CENTRAL
    ) @ finite_difference_jacobian(G, x, 1e-6, Difference.CENTRAL)
