"""Exercises 3-6: binomial and lognormal moments, calibrating a binomial tree.

3. E[S(t+dt)] = (p u + (1-p) d) S, E[S^2(t+dt)] = (p u^2 + (1-p) d^2) S^2 (two-point expectation).
4. E[e^(aZ)] = e^(a^2/2)  =>  E[S(t+dt)] = e^(r dt) S, E[S^2(t+dt)] = e^((2r + sigma^2) dt) S^2.
5. (i) (4.57) <=> p = (e^(r dt) - d)/(u - d); then (4.58) <=> e^(r dt)(u + d) - u d = e^((2r+sigma^2) dt)
       <=> (e^(r dt) - d)(u - e^(r dt)) = e^(2r dt)(e^(sigma^2 dt) - 1).
   (ii) u d = 1: u + 1/u = 2A, A = (e^(-r dt) + e^((r + sigma^2) dt))/2, u = A + sqrt(A^2 - 1), d = 1/u.
6. p = 1/2: u, d = e^(r dt)(1 +/- sqrt(e^(sigma^2 dt) - 1)).

Full derivations: cpp/ch04_lognormal_risk_neutral/exercises/include/p4mfe/ch04/exercises/ex03_06_binomial_calibration.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def ex03_moment(k: int, S: float, u: float, d: float, p: float) -> float:
    return p * (S * u) ** k + (1.0 - p) * (S * d) ** k


def ex04_first_moment(S: float, r: float, dt: float) -> float:
    return S * math.exp(r * dt)


def ex04_second_moment(S: float, r: float, sigma: float, dt: float) -> float:
    return S * S * math.exp((2.0 * r + sigma * sigma) * dt)


@dataclass(frozen=True)
class BinomialStep:
    u: float
    d: float
    p: float


def ex05_crr(r: float, sigma: float, dt: float) -> BinomialStep:
    A = 0.5 * (math.exp(-r * dt) + math.exp((r + sigma * sigma) * dt))
    root = math.sqrt(A * A - 1.0)
    u, d = A + root, A - root
    return BinomialStep(u, d, (math.exp(r * dt) - d) / (u - d))


def moment_residuals(s: BinomialStep, r: float, sigma: float, dt: float) -> tuple[float, float]:
    """Residuals of (4.57) and (4.58)."""
    return (s.p * s.u + (1.0 - s.p) * s.d - math.exp(r * dt),
            s.p * s.u ** 2 + (1.0 - s.p) * s.d ** 2 - math.exp((2.0 * r + sigma * sigma) * dt))


def ex05_residual_460(u: float, d: float, r: float, sigma: float, dt: float) -> float:
    g = math.exp(r * dt)
    return (g - d) * (u - g) - g * g * math.expm1(sigma * sigma * dt)


def ex06_equal_probability(r: float, sigma: float, dt: float) -> BinomialStep:
    x = math.sqrt(math.expm1(sigma * sigma * dt))
    if x >= 1.0:
        raise ValueError("ex06_equal_probability: need sigma^2 dt < ln 2")
    g = math.exp(r * dt)
    return BinomialStep(g * (1.0 + x), g * (1.0 - x), 0.5)
