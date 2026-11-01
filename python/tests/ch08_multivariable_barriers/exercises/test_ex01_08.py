import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.integration import simpson_rule
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_d1_d2
from p4mfe.ch08_multivariable_barriers.double_integrals import jacobian_determinant
from p4mfe.ch08_multivariable_barriers.exercises.ex01_gamma_maximum import (
    Ex01Market, ex01_argmax, ex01_gamma, ex01_log_derivative, ex01_max_gamma)
from p4mfe.ch08_multivariable_barriers.exercises.ex02_03_double_integrals import (
    EX02_ABS_EXACT, EX03_EXACT, ex02_abs_direct, ex02_change_of_variables, ex02_direct, ex03_inner, ex03_numeric)
from p4mfe.ch08_multivariable_barriers.exercises.ex04_05_polar_areas import (
    ex04_disk_area_cartesian, ex04_disk_area_polar, ex05_ellipse_area, ex05_exact, ex05_map)
from p4mfe.ch08_multivariable_barriers.exercises.ex06_07_exp_inequalities import (
    ex06_e_pow_pi, ex06_f, ex06_pi_pow_e, ex07_gap, ex07_lhs, ex07_rhs)
from p4mfe.ch08_multivariable_barriers.exercises.ex08_gronwall import ex08_check, ex08_v, ex08_V


@pytest.mark.parametrize("m", [Ex01Market(), Ex01Market(100.0, 2.0, 0.5, 0.02), Ex01Market(40.0, 0.1, 0.15, 0.0)])
def test_ex01_gamma_rises_then_falls(m):
    s = ex01_argmax(m)
    assert ex01_gamma(s, m) == pytest.approx(ex01_max_gamma(m), rel=1e-12)
    assert bs_d1_d2(s, m.K, m.T, m.sigma, m.r)[0] == pytest.approx(-m.sigma * math.sqrt(m.T), abs=1e-12)
    up = [ex01_gamma(S, m) for S in np.linspace(0.2 * s, s, 81)]
    down = [ex01_gamma(S, m) for S in np.linspace(s, 5 * s, 401)]
    assert np.all(np.diff(up) > 0) and np.all(np.diff(down) < 0)
    assert ex01_log_derivative(0.9 * s, m) > 0 > ex01_log_derivative(1.1 * s, m)
    grid = m.K * np.exp(np.linspace(math.log(1e-3), math.log(1e3), 300))
    assert max(ex01_gamma(S, m) for S in grid) <= ex01_max_gamma(m)
    assert ex01_gamma(1e-3 * m.K, m) < 1e-6 * ex01_max_gamma(m)
    assert ex01_gamma(1e3 * m.K, m) < 1e-6 * ex01_max_gamma(m)


def test_ex02_integral_is_zero():
    assert ex02_direct() == pytest.approx(0.0, abs=1e-6)
    assert ex02_change_of_variables() == pytest.approx(0.0, abs=1e-12)
    assert ex02_abs_direct(400) == pytest.approx(EX02_ABS_EXACT, abs=1e-3)


def test_ex03_parabola_and_line():
    assert ex03_numeric() == pytest.approx(EX03_EXACT, abs=1e-10)
    for x in (1.0, 1.3, 1.7, 2.0):
        lo, hi = (x + 1) ** 2, 5 * x - 1
        assert (x * x * hi - hi * hi) - (x * x * lo - lo * lo) == pytest.approx(ex03_inner(x), abs=1e-12)
    assert simpson_rule(ex03_inner, 1.0, 2.0, 2) == pytest.approx(-1.75, abs=1e-14)


@pytest.mark.parametrize("R", [0.5, 1.0, 3.0])
def test_ex04_disk_area(R):
    assert ex04_disk_area_polar(R) == pytest.approx(math.pi * R * R, rel=1e-10)
    assert ex04_disk_area_cartesian(R) == pytest.approx(math.pi * R * R, rel=1e-4)


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (3.0, 0.5), (0.2, 7.0)])
def test_ex05_ellipse_area(a, b):
    assert ex05_ellipse_area(a, b) == pytest.approx(ex05_exact(a, b), rel=1e-6)
    assert jacobian_determinant(lambda r, t: ex05_map(a, b, r, t), 0.7, 1.1) == pytest.approx(a * b * 0.7, rel=1e-7)


def test_ex06():
    assert ex06_e_pow_pi() > ex06_pi_pow_e()
    assert ex06_e_pow_pi() == pytest.approx(23.140693, abs=1e-6)
    assert ex06_pi_pow_e() == pytest.approx(22.459158, abs=1e-6)
    for x in np.arange(0.5, 20.0, 0.05):
        if abs(x - math.e) > 1e-9:
            assert ex06_f(x) < 1 / math.e


def test_ex07():
    for x in np.arange(-5.0, 5.01, 0.25):
        for y in np.arange(-5.0, 5.01, 0.25):
            assert ex07_lhs(x, y) >= ex07_rhs(x, y) * (1 - 1e-15)
            assert ex07_lhs(x, y) - ex07_rhs(x, y) == pytest.approx(ex07_gap(x, y), abs=1e-13 * ex07_lhs(x, y))
    assert ex07_gap(1.3, 1.3) == 0.0
    assert ex07_gap(1.3, 1.3 + 1e-6) > 0.0


def test_ex08_gronwall():
    M, X = 2.0, 6.0
    c1 = ex08_check(lambda x: M * math.exp(0.5 * ex08_V(x)), ex08_v, M, X)
    assert c1.hypothesis_gap <= 0.0 and c1.conclusion_gap <= 1e-12 and c1.g_increase < 1e-6
    c2 = ex08_check(lambda x: M * (1 + 0.5 * ex08_V(x)), ex08_v, M, X)
    assert c2.hypothesis_gap <= 1e-9 and c2.conclusion_gap <= 1e-12 and c2.g_increase < 1e-6
    c3 = ex08_check(lambda x: M * math.exp(ex08_V(x)), ex08_v, M, X)
    assert abs(c3.conclusion_gap) < 1e-5
    assert abs(c3.hypothesis_gap) < 1e-3 * M * math.exp(ex08_V(X))
    assert c3.g_increase < 1e-5
