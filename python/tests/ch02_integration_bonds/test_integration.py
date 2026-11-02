import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.integration import (
    Rule,
    gamma_function,
    improper_integral,
    improper_integral_derivative,
    integrate_to_tolerance,
    midpoint_rule,
    observed_order,
    simpson_rule,
    trapezoidal_rule,
)

gauss = lambda x: np.exp(-x * x)
GAUSS_0_2 = 0.5 * math.sqrt(math.pi) * math.erf(2.0)  # 2.4.2 example
inf = math.inf


def test_exact_for_low_degree_polynomials():
    line = lambda x: 3 * x - 1
    cubic = lambda x: x**3 - 2 * x**2 + x + 5
    assert midpoint_rule(line, 0, 2, 1) == pytest.approx(4.0, abs=1e-14)
    assert trapezoidal_rule(line, 0, 2, 1) == pytest.approx(4.0, abs=1e-14)
    assert simpson_rule(cubic, 0, 2, 1) == pytest.approx(32 / 3, abs=1e-13)


@pytest.mark.parametrize("n", [1, 4, 17])
def test_simpson_is_weighted_trapezoidal_and_midpoint(n):
    expected = (trapezoidal_rule(gauss, 0, 2, n) + 2 * midpoint_rule(gauss, 0, 2, n)) / 3
    assert simpson_rule(gauss, 0, 2, n) == pytest.approx(expected, abs=1e-14)


def test_midpoint_and_trapezoidal_bracket_convex_integrand():
    exact = math.e - 1
    assert midpoint_rule(np.exp, 0, 1, 8) < exact < trapezoidal_rule(np.exp, 0, 1, 8)
    ratio = (exact - midpoint_rule(np.exp, 0, 1, 64)) / (trapezoidal_rule(np.exp, 0, 1, 64) - exact)
    assert ratio == pytest.approx(0.5, abs=1e-3)


def test_observed_orders():
    assert observed_order(gauss, 0, 2, GAUSS_0_2, 32, Rule.MIDPOINT) == pytest.approx(2, abs=0.05)
    assert observed_order(gauss, 0, 2, GAUSS_0_2, 32, Rule.TRAPEZOIDAL) == pytest.approx(2, abs=0.05)
    assert observed_order(gauss, 0, 2, GAUSS_0_2, 16, Rule.SIMPSON) == pytest.approx(4, abs=0.1)


@pytest.mark.parametrize("rule", list(Rule))
def test_concrete_example_to_tolerance(rule):
    assert integrate_to_tolerance(gauss, 0, 2, 1e-7, rule).value == pytest.approx(GAUSS_0_2, abs=1e-7)


def test_simpson_needs_fewer_intervals():
    simpson = integrate_to_tolerance(gauss, 0, 2, 1e-7, Rule.SIMPSON).intervals
    midpoint = integrate_to_tolerance(gauss, 0, 2, 1e-7, Rule.MIDPOINT).intervals
    assert simpson < midpoint / 8


def test_improper_infinite_limits():
    assert improper_integral(lambda x: 1 / (x * x), 1, inf).value == pytest.approx(1.0, abs=1e-12)
    assert improper_integral(lambda x: np.exp(-x), 0, inf).value == pytest.approx(1.0, abs=1e-8)
    assert improper_integral(lambda x: np.exp(-x * x / 2), -inf, inf).value == pytest.approx(
        math.sqrt(2 * math.pi), abs=1e-8
    )
    assert improper_integral(np.exp, -inf, 0).value == pytest.approx(1.0, abs=1e-8)
    assert improper_integral(lambda x: 1 / (1 + x * x), -inf, inf).value == pytest.approx(math.pi, abs=1e-8)
    with pytest.raises(ValueError):
        improper_integral(gauss, 1, 0)


def test_gamma_function():
    assert gamma_function(1) == pytest.approx(1.0, abs=1e-8)
    assert gamma_function(5) == pytest.approx(24.0, abs=1e-7)
    assert gamma_function(2.5) == pytest.approx(math.gamma(2.5), abs=1e-8)
    assert gamma_function(4.3) == pytest.approx(3.3 * gamma_function(3.3), abs=1e-7)
    assert gamma_function(0.5) == pytest.approx(math.sqrt(math.pi), abs=1e-8)
    assert gamma_function(1.5) == pytest.approx(0.5 * math.sqrt(math.pi), abs=1e-8)
    assert gamma_function(0.3) == pytest.approx(math.gamma(0.3), abs=1e-7)


def test_differentiate_under_integral_sign():
    t = 1.7
    got = improper_integral_derivative(lambda x, s: np.exp(-s * x), 0, inf, t)
    assert got == pytest.approx(-1 / t**2, abs=1e-7)
    expected = -t / 2 * math.sqrt(math.pi) / 2 * math.exp(-t * t / 4)
    got = improper_integral_derivative(lambda x, s: np.exp(-x * x) * np.cos(s * x), 0, inf, t)
    assert got == pytest.approx(expected, abs=1e-7)
