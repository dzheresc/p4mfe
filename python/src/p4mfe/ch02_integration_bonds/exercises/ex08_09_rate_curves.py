"""Exercises 8-9: parallel shifts and compounding conventions of rate curves.

Exercise 8. t r(0,t) = int_0^t r(s) ds. If r2(0,t) = r1(0,t) + dr, differentiate
    t r2(0,t) = t r1(0,t) + t dr to get r2(t) = r1(t) + dr; conversely integrate
    r2 = r1 + dr over [0, t] and divide by t.

Exercise 9. r_c(0,t) = 0.05 + 0.01 ln(1 + t/2).
    (i)   r(t) = d/dt (t r_c(0,t)) = 0.05 + 0.01 ln(1 + t/2) + 0.01 t/(2 + t).
    (ii)  annual:      r_1(0,t) = e^(r_c(0,t)) - 1 = e^0.05 (1 + t/2)^0.01 - 1.
    (iii) semiannual:  r_2(0,t) = 2 (e^(r_c(0,t)/2) - 1) = 2 (e^0.025 (1 + t/2)^0.005 - 1).
"""

from __future__ import annotations

import numpy as np


def ex08_shift(zero, dr: float):
    return lambda t: zero(t) + dr


def ex09_zero_rate(t):
    return 0.05 + 0.01 * np.log1p(0.5 * t)


def ex09_inst_rate(t):
    return ex09_zero_rate(t) + 0.01 * t / (2 + t)


def ex09_annual_zero_rate(t):
    return np.exp(0.05) * (1 + 0.5 * t) ** 0.01 - 1


def ex09_semiannual_zero_rate(t):
    return 2 * (np.exp(0.025) * (1 + 0.5 * t) ** 0.005 - 1)
