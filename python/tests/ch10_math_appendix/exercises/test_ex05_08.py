import math

import pytest

from p4mfe.ch10_math_appendix.exercises.ex05_07_linear_recursions import (
    EX05_INIT,
    EX05_RECURSION,
    EX07_INIT,
    EX07_RECURSION,
    PHI,
    PSI,
    ex05_basis,
    ex05_closed_form,
    ex05_constants,
    ex06_closed_form,
    ex06_first_order,
    ex06_second_order,
    ex07_closed_form,
    ex07_constants,
    ex07_direct,
    ex_evaluate,
    ex_fit_constants,
)
from p4mfe.ch10_math_appendix.exercises.ex08_double_root import ex08_hint, ex08_residual
from p4mfe.ch10_math_appendix.linear_recursions import recursion_terms


def test_ex05_alternating_sign():
    x = recursion_terms(EX05_RECURSION, EX05_INIT, 80)
    assert x == [ex05_closed_form(n) for n in range(80)]
    assert list(ex05_constants()) == pytest.approx([1, 0, 0], abs=1e-12)
    for z in (-1.0, PHI, PSI):
        assert z**3 - 2 * z - 1 == pytest.approx(0, abs=1e-12)
    init = (2.0, 0.5, -1.0)
    c = ex_fit_constants(ex05_basis(), init)
    y = recursion_terms(EX05_RECURSION, init, 30)
    for n, yn in enumerate(y):
        assert ex_evaluate(ex05_basis(), c, n) == pytest.approx(
            yn, rel=1e-9, abs=1e-9
        )


def test_ex06_closed_form():
    x = recursion_terms([-3.0, 4.0], [1.0, 5.0], 30)
    r = ex06_second_order()
    direct = 1.0
    for n in range(30):
        assert x[n] == ex06_closed_form(n) == direct
        assert r(n) == pytest.approx(direct, rel=1e-12)
        assert ex06_first_order(n) == pytest.approx(direct, rel=1e-12)
        direct = 3 * direct + 2
    assert sorted([r.lambda1.real, r.lambda2.real]) == pytest.approx([1.0, 3.0])


def test_ex07_closed_form():
    d = ex07_direct(30)
    assert d[:3] == [1.0, 5.0, 18.0]
    x = recursion_terms(EX07_RECURSION, EX07_INIT, 30)
    for n in range(30):
        assert x[n] == d[n] == ex07_closed_form(n)
    assert list(ex07_constants()) == pytest.approx(
        [-1.25, -0.5, 2.25], abs=1e-12
    )


def test_ex08_double_root():
    a = [4.0, 0.0, -3.0, 1.0]  # (z - 2)^2 (z + 1)
    for n in range(21):
        scale = 2.0 ** (n + 3) * (n + 3)
        assert abs(ex08_residual(a, 2.0, 1.7, n)) <= 1e-13 * scale
        assert abs(ex08_hint(a, 2.0, 1.7, n)) <= 1e-13 * scale
        r = ex08_residual(a, -1.0, 1.7, n)
        assert r == pytest.approx(ex08_hint(a, -1.0, 1.7, n), abs=1e-12)
        assert abs(r) == pytest.approx(1.7 * 9, abs=1e-12)
    b = [0.5, -1.2, 0.3, 2.0]
    for n in range(11):
        assert ex08_residual(b, 0.8, -2.0, n) == pytest.approx(
            ex08_hint(b, 0.8, -2.0, n), abs=1e-12
        )
    x = recursion_terms([-4.0, 0.0, 3.0], [0.0, 2.0, 8.0], 40)
    assert x == [n * 2.0**n for n in range(40)]
