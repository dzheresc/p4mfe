"""Exercise 16: u(x, t) = e^(-x^2/(4t)) / sqrt(4 pi t).

u_t = u (x^2 - 2t) / (4t^2) = u_xx, so u solves the heat equation.
"""

from __future__ import annotations

import numpy as np


def ex16_u(x, t):
    return np.exp(-x * x / (4.0 * t)) / np.sqrt(4.0 * np.pi * t)


def ex16_u_t(x, t):
    return ex16_u(x, t) * (x * x - 2.0 * t) / (4.0 * t * t)


def ex16_u_x(x, t):
    return ex16_u(x, t) * (-x / (2.0 * t))


def ex16_u_xx(x, t):
    return ex16_u(x, t) * (x * x - 2.0 * t) / (4.0 * t * t)
