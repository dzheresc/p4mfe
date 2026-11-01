"""Exercise 7: I_n^S = (2/3) I_n^M + (1/3) I_n^T.

On each interval [a_{i-1}, a_i] with midpoint m_i,
    h (f(a_{i-1}) + 4 f(m_i) + f(a_i)) / 6 = (2/3) h f(m_i) + (1/3) h (f(a_{i-1}) + f(a_i)) / 2;
sum over i. The weights cancel the h^2 error terms (+h^2/24 and -h^2/12 times int f'').
"""

from __future__ import annotations

from p4mfe.ch02_integration_bonds.integration import midpoint_rule, trapezoidal_rule


def ex07_simpson_from_mt(f, a: float, b: float, n: int) -> float:
    return (2 * midpoint_rule(f, a, b, n) + trapezoidal_rule(f, a, b, n)) / 3
