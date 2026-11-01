"""Exercise 16: B = 101, D = 1.5, C = 2.5; price after dy = 10bp, 50bp, 1%, 2% by
(2.57) B (1 - D dy) and (6.73) B (1 - D dy + C dy^2/2).

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex16_bond_price_change.hpp
"""

from __future__ import annotations

from dataclasses import dataclass

from p4mfe.ch02_integration_bonds.bonds import price_approx_duration, price_approx_duration_convexity

EX16_YIELD_CHANGES = (0.001, 0.005, 0.01, 0.02)


@dataclass(frozen=True)
class Ex16Row:
    dy: float
    duration_only: float
    duration_convexity: float


def ex16_table(B: float = 101.0, D: float = 1.5, C: float = 2.5) -> list[Ex16Row]:
    return [
        Ex16Row(dy, price_approx_duration(B, D, dy), price_approx_duration_convexity(B, D, C, dy))
        for dy in EX16_YIELD_CHANGES
    ]
