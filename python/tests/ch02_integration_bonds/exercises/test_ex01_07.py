import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.exercises.ex01_02_improper_integrals import (
    EX01_LIMIT,
    ex01_quotient,
    ex02_gamma,
    ex02_left_bound,
    ex02_left_piece,
)
from p4mfe.ch02_integration_bonds.exercises.ex03_04_quadrature import (
    EX04_SIMPSON_ORDER,
    ex03_approx,
    ex03_exact,
    ex04_exact,
    ex04_f,
    ex04_f4,
    ex04_midpoint,
    ex04_simpson_table,
)
from p4mfe.ch02_integration_bonds.exercises.ex05_06_integral_derivatives import (
    Ex05Params,
    Ex06Market,
    ex05_b,
    ex05_g,
    ex05_g_prime,
    ex06_call,
    ex06_discounted_density,
    ex06_g,
    ex06_g_normal,
    ex06_g_normal_prime,
)
from p4mfe.ch02_integration_bonds.exercises.ex07_simpson_identity import ex07_simpson_from_mt
from p4mfe.ch02_integration_bonds.integration import Rule, improper_integral, integrate_to_tolerance, simpson_rule
from p4mfe.normal import norm_cdf, norm_pdf


def _phi(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def test_ex01_limit():
    assert EX01_LIMIT == pytest.approx(-0.0664903800669, abs=1e-12)
    for x in (0.5, 0.1, 1e-2, 1e-4, 1e-8, -1e-3):
        assert ex01_quotient(x) == pytest.approx(
            EX01_LIMIT, abs=x * x / 40 + 1e-16
        )
    x = 0.9
    naive = (norm_cdf(x) - 0.5 - x / math.sqrt(2 * math.pi)) / x**3
    assert ex01_quotient(x) == pytest.approx(naive, abs=1e-13)


@pytest.mark.parametrize("a", [0.25, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0])
def test_ex02_gamma(a):
    assert ex02_left_piece(a) < ex02_left_bound(a)
    assert ex02_gamma(a) == pytest.approx(math.gamma(a), rel=1e-8)


def test_ex02_recursion_and_factorial():
    for a in (1.25, 2.5, 3.0, 4.75):
        assert ex02_gamma(a) == pytest.approx(
            (a - 1) * ex02_gamma(a - 1), abs=1e-7
        )
    for n in range(1, 9):
        assert ex02_gamma(n) == pytest.approx(math.factorial(n - 1), rel=1e-9)


def test_ex03():
    exact = ex03_exact()
    assert exact == pytest.approx(0.4083702471769958, abs=1e-14)
    m, t, s = (
        ex03_approx(r) for r in (Rule.MIDPOINT, Rule.TRAPEZOIDAL, Rule.SIMPSON)
    )
    assert (m.intervals, t.intervals, s.intervals) == (256, 512, 16)
    assert m.value == pytest.approx(0.408369962, abs=1e-9)
    assert t.value == pytest.approx(0.408370390, abs=1e-9)
    assert s.value == pytest.approx(0.408370194, abs=1e-9)
    assert m.value < exact < t.value
    for r in (m, t, s):
        assert abs(r.value - exact) < 1e-6


def test_ex04_midpoint_and_exact():
    I = ex04_exact()
    assert I == pytest.approx(0.1791711722673056, abs=1e-15)
    assert I == pytest.approx(
        integrate_to_tolerance(
            lambda u: 2 * u**6 / (1 + u**4), 0, 1, 1e-14
        ).value,
        abs=1e-13,
    )
    m = ex04_midpoint()
    assert m.intervals == 512
    assert m.value == pytest.approx(I, abs=1e-6)


def test_ex04_fourth_derivative_unbounded():
    x, h = 0.4, 1e-2
    fd = (
        ex04_f(x + 2 * h)
        - 4 * ex04_f(x + h)
        + 6 * ex04_f(x)
        - 4 * ex04_f(x - h)
        + ex04_f(x - 2 * h)
    ) / h**4
    assert ex04_f4(x) == pytest.approx(fd, rel=1e-3, abs=1e-2)
    for z in (1e-2, 1e-4, 1e-6, 1e-8):
        assert ex04_f4(z) / (-0.9375 * z**-1.5) == pytest.approx(1, abs=20 * z)
    assert ex04_f4(1e-8) < -1e11


def test_ex04_simpson_order():
    rows = ex04_simpson_table()
    assert [r.n for r in rows] == [4, 8, 16, 32, 64, 128, 256]
    for prev, cur in zip(rows[1:], rows[2:]):
        assert math.log2(prev.error / cur.error) == pytest.approx(
            EX04_SIMPSON_ORDER, abs=0.05
        )


def test_ex05_call_gamma():
    p = Ex05Params(100.0, 0.5, 0.3, 0.04)
    for x in (60.0, 90.0, 100.0, 115.0, 160.0):
        h = 1e-4 * x
        assert ex05_g_prime(p, x) == pytest.approx(
            (ex05_g(p, x + h) - ex05_g(p, x - h)) / (2 * h), abs=1e-9
        )
    gi = improper_integral(_phi, -math.inf, ex05_b(p, 105.0), 1e-11).value
    assert gi == pytest.approx(ex05_g(p, 105.0), abs=1e-9)


@pytest.mark.parametrize("t", [-2.0, -0.5, 0.0, 0.7, 2.5])
def test_ex06_normal(t):
    assert ex06_g(_phi, t) == pytest.approx(ex06_g_normal(t), abs=1e-9)
    h = 1e-3
    assert (ex06_g_normal(t + h) - ex06_g_normal(t - h)) / (
        2 * h
    ) == pytest.approx(ex06_g_normal_prime(t), abs=1e-7)
    g2 = (
        ex06_g_normal(t + h) - 2 * ex06_g_normal(t) + ex06_g_normal(t - h)
    ) / h**2
    assert g2 == pytest.approx(norm_pdf(t), abs=1e-6)


def test_ex06_polynomial_decay():
    hh = lambda x: 1 / (1 + x * x) ** 2  # noqa: E731
    t, h = 0.3, 1e-2
    g2 = (
        ex06_g(hh, t + h, 1e-12)
        - 2 * ex06_g(hh, t, 1e-12)
        + ex06_g(hh, t - h, 1e-12)
    ) / h**2
    assert g2 == pytest.approx(hh(t), abs=1e-4)


def test_ex06_call_density():
    m = Ex06Market(100.0, 1.0, 0.25, 0.03)
    for K in (70.0, 95.0, 100.0, 120.0, 150.0):
        h = 1e-2 * K
        c2 = (
            ex06_call(m, K + h) - 2 * ex06_call(m, K) + ex06_call(m, K - h)
        ) / h**2
        assert c2 == pytest.approx(ex06_discounted_density(m, K), abs=1e-5)
    K = 110.0
    assert ex06_g(
        lambda x: ex06_discounted_density(m, x), K, 1e-11
    ) == pytest.approx(ex06_call(m, K), abs=1e-7)


def test_ex07_identity():
    fs = [
        lambda x: np.exp(-x * x),
        lambda x: np.sqrt(x) * np.exp(-x),
        lambda x: np.sin(5 * x) / (1 + x),
    ]
    for f in fs:
        for n in (1, 2, 3, 7, 64):
            assert ex07_simpson_from_mt(f, 0.5, 2.5, n) == pytest.approx(
                simpson_rule(f, 0.5, 2.5, n), abs=1e-14
            )
