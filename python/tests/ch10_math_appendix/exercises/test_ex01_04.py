import math

import pytest

from p4mfe.ch10_math_appendix.exercises.ex01_parity import ex01_h, ex01_predicted_parity
from p4mfe.ch10_math_appendix.exercises.ex02_lhopital_sums import (
    ex02_book_n2,
    ex02_book_n3,
    ex02_derivative_at_one,
    ex02_numerator,
    ex02_sum_by_lhopital,
    ex02_t_closed_form,
)
from p4mfe.ch10_math_appendix.exercises.ex03_sum_k4 import ex03_sum_k4, ex03_sum_k4_recursive
from p4mfe.ch10_math_appendix.exercises.ex04_sum_squares_recursion import (
    EX04_RECURSION,
    ex04_cubic,
    ex04_terms,
    ex04_terms_10_93,
)
from p4mfe.ch10_math_appendix.interpolation import polynomial_value
from p4mfe.ch10_math_appendix.parity import is_even, is_odd, parity
from p4mfe.ch10_math_appendix.sums import sum_k, sum_k2, sum_k3, sum_kj_xk, sum_of_powers, sum_of_powers_polynomial

XS = [0.1, 0.37, 0.8, 1.0, 1.3, 1.9, 2.4]
ODD = [math.sin, lambda x: x**3 - 2 * x, lambda x: math.atan(3 * x)]


@pytest.mark.parametrize("f", ODD)
def test_ex01_parts_one_and_two(f):
    assert is_odd(f, XS)
    assert is_even(lambda x: x * f(x), XS)
    assert is_odd(lambda x: x * x * f(x), XS)
    assert is_even(lambda x: f(x * x), XS)
    assert is_odd(lambda x: f(x**3), XS)


@pytest.mark.parametrize("f", ODD)
def test_ex01_general_rule(f):
    for i in range(1, 5):
        for j in range(1, 5):
            assert parity(lambda x: ex01_h(f, i, j, x), XS, 1e-10) == ex01_predicted_parity(i, j)


def test_ex02_numerators_match_the_book():
    for n in range(3, 31):
        assert ex02_numerator(n, 2) == ex02_book_n2(n)
        assert ex02_numerator(n, 3) == ex02_book_n3(n)


def test_ex02_closed_forms():
    for n in (1, 2, 5, 12):
        for x in (-0.7, 0.3, 0.9, 1.1, 2.0):
            for j in range(4):
                direct = sum_kj_xk(n, j, x)
                assert ex02_t_closed_form(n, j, x) == pytest.approx(direct, rel=1e-9, abs=1e-9)


def test_ex02_lhopital_gives_the_sums():
    for n in range(1, 201):
        for j in (2, 3):
            N = ex02_numerator(n, j)
            assert all(ex02_derivative_at_one(N, r) == 0 for r in range(j + 1))
        assert ex02_derivative_at_one(ex02_numerator(n, 2), 3) == -n * (n + 1) * (2 * n + 1)
        assert ex02_sum_by_lhopital(n, 2) == sum_k2(n)
        assert ex02_sum_by_lhopital(n, 3) == sum_k3(n)
    for n in range(1, 51):
        assert ex02_sum_by_lhopital(n, 1) == sum_k(n)
        assert ex02_sum_by_lhopital(n, 4) == sum_of_powers(n, 4)
    # Python integers: no overflow for large n.
    n = 10**6
    assert ex02_sum_by_lhopital(n, 3) == sum_k3(n)


def test_ex03_sum_k4():
    for n in range(301):
        assert ex03_sum_k4(n) == sum_of_powers(n, 4) == ex03_sum_k4_recursive(n)
    assert ex03_sum_k4(10**9) == ex03_sum_k4_recursive(10**9)
    assert (ex03_sum_k4(1), ex03_sum_k4(2), ex03_sum_k4(10)) == (1, 17, 25333)
    c = sum_of_powers_polynomial(4)
    assert list(c) == pytest.approx([0, -1 / 30, 0, 1 / 3, 0.5, 0.2], abs=1e-9)


def test_ex04_recursions_reproduce_sum_of_squares():
    x = ex04_terms(60)
    y = ex04_terms_10_93(60)
    for i in range(60):
        assert x[i] == y[i] == sum_k2(i + 1)
    assert list(ex04_cubic()) == pytest.approx([0, 1 / 6, 0.5, 1 / 3], abs=1e-12)
    P = list(EX04_RECURSION) + [-1.0]
    for z in (-1.5, 0.2, 2.0, 3.0):
        assert -polynomial_value(P, z) == pytest.approx((z - 1) ** 4, abs=1e-12)
