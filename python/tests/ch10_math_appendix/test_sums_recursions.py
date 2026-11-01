import math

import pytest

from p4mfe.ch10_math_appendix.interpolation import polynomial_value
from p4mfe.ch10_math_appendix.linear_recursions import (
    first_order_affine,
    loan_payment,
    recursion_terms,
    solve_second_order,
)
from p4mfe.ch10_math_appendix.sums import (
    geometric_sum,
    kj_xk_numerator,
    sum_k,
    sum_k2,
    sum_k2_xk,
    sum_k3,
    sum_k_xk,
    sum_kj_xk,
    sum_kj_xk_infinite,
    sum_of_powers,
    sum_of_powers_polynomial,
    sums_of_powers_recursive,
)


# 10.3.1 --------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 7, 50, 1000])
def test_closed_forms(n):
    assert sum_k(n) == sum_of_powers(n, 1)
    assert sum_k2(n) == sum_of_powers(n, 2)
    assert sum_k3(n) == sum_of_powers(n, 3)


@pytest.mark.parametrize("n", [1, 5, 20, 100, 10**6])
def test_telescoping_recursion(n):
    S = sums_of_powers_recursive(n, 6)
    if n <= 100:
        assert S == [sum_of_powers(n, i) for i in range(7)]
    assert S[2] == sum_k2(n) and S[3] == sum_k3(n)  # exact for huge n too


def test_sum_of_powers_polynomial():
    assert sum_of_powers_polynomial(2) == pytest.approx([0, 1 / 6, 1 / 2, 1 / 3], abs=1e-12)
    for i in range(1, 7):
        p = sum_of_powers_polynomial(i)
        assert p[-1] == pytest.approx(1 / (i + 1), abs=1e-9)
        assert p[i] == pytest.approx(0.5, abs=1e-9)
        assert polynomial_value(p, 30.0) == pytest.approx(sum_of_powers(30, i), rel=1e-9)


# 10.3.2 --------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 4, 25])
@pytest.mark.parametrize("x", [-0.8, 0.3, 0.95, 1.0, 1.7])
def test_finite_sums_kj_xk(n, x):
    tol = 1e-11 * max(1.0, abs(sum_kj_xk(n, 2, x)))
    assert geometric_sum(n, x) == pytest.approx(sum_kj_xk(n, 0, x), abs=tol)
    assert sum_k_xk(n, x) == pytest.approx(sum_kj_xk(n, 1, x), abs=tol)
    assert sum_k2_xk(n, x) == pytest.approx(sum_kj_xk(n, 2, x), abs=tol)


def test_infinite_sums_kj_xk():
    assert kj_xk_numerator(0) == [0, 1]
    assert kj_xk_numerator(3) == [0, 1, 4, 1]
    assert kj_xk_numerator(4) == [0, 1, 11, 11, 1]
    for x in [-0.5, 0.2, 0.9]:
        assert sum_kj_xk_infinite(1, x) == pytest.approx(x / (1 - x) ** 2, rel=1e-13)
        assert sum_kj_xk_infinite(2, x) == pytest.approx(x * (1 + x) / (1 - x) ** 3, rel=1e-12)
        for j in range(6):
            assert sum_kj_xk(2000, j, x) == pytest.approx(sum_kj_xk_infinite(j, x), rel=1e-10)
    with pytest.raises(ValueError):
        sum_kj_xk_infinite(1, 1.0)


def test_perpetuity_duration():
    x = math.exp(-0.05)
    assert sum_kj_xk_infinite(1, x) / sum_kj_xk_infinite(0, x) == pytest.approx(1 / (1 - x), rel=1e-12)


# 10.4 ----------------------------------------------------------------------
def test_fibonacci_closed_form():
    fib = recursion_terms([1.0, 1.0], [0.0, 1.0], 40)
    assert fib[10] == 55
    r = solve_second_order(1.0, 1.0, 0.0, 1.0)
    assert not r.double_root
    assert r.lambda1.real == pytest.approx((1 + math.sqrt(5)) / 2, abs=1e-15)
    for n in range(40):
        assert r(n) == pytest.approx(fib[n], rel=1e-9, abs=1e-9)


def test_double_and_complex_roots():
    d = solve_second_order(4.0, -4.0, 1.0, 4.0)
    assert d.double_root
    td = recursion_terms([-4.0, 4.0], [1.0, 4.0], 20)
    assert [d(n) for n in range(20)] == pytest.approx(td, rel=1e-9)
    c = solve_second_order(1.0, -1.0, 1.0, 2.0)
    tc = recursion_terms([-1.0, 1.0], [1.0, 2.0], 30)
    assert [c(n) for n in range(30)] == pytest.approx(tc, abs=1e-12)
    assert tc[6] == tc[0]
    assert abs(c.lambda1) == pytest.approx(1.0, abs=1e-15)


def test_third_order():
    t = recursion_terms([6.0, -11.0, 6.0], [3.0, 6.0, 14.0], 15)
    assert t == [1 + 2**n + 3**n for n in range(15)]


def test_affine_and_loan():
    x = 5.0
    for n in range(12):
        assert first_order_affine(0.7, 3.0, 5.0, n) == pytest.approx(x, abs=1e-12)
        x = 0.7 * x + 3.0
    assert first_order_affine(1.0, 2.0, 1.0, 5) == 11.0
    P = loan_payment(300000.0, 0.005, 360)
    assert P == pytest.approx(1798.65, abs=0.01)
    assert first_order_affine(1.005, -P, 300000.0, 360) == pytest.approx(0.0, abs=1e-6)
