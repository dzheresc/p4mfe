import math

import pytest

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call
from p4mfe.ch08_multivariable_barriers.exercises.ex09_10_heat_change_of_variables import (
    ex09_call_u0,
    ex09_coefficients,
    ex09_heat_residual,
    ex09_put_u0,
    ex09_u,
    ex10_printed_b,
    ex10_residuals,
    ex10_solve,
)
from p4mfe.ch08_multivariable_barriers.exercises.ex11_barrier_rebate import (
    ex11_down_and_out_call_with_rebate,
    ex11_heat_fd,
    ex11_rebate_value,
    ex11_u_boundary,
)
from p4mfe.ch08_multivariable_barriers.exercises.ex12_13_asian_pdes import (
    AsianMarket,
    ex12_f_residual,
    ex12_geometric_asian_call,
    ex12_v_residual,
    ex12_y,
    ex13_h_linear,
    ex13_h_residual,
    ex13_v_residual,
)
from p4mfe.ch08_multivariable_barriers.heat_equation import heat_transform


def test_ex09_initial_conditions():
    strike, sigma, r, q = 50.0, 0.3, 0.05, 0.02
    c = ex09_coefficients(sigma, r, q)
    for i in range(21):
        x = -1.0 + 0.1 * i
        assert ex09_u(
            OptionType.CALL, x, 0.0, strike, sigma, r, q
        ) == pytest.approx(ex09_call_u0(x, strike, c.a), abs=1e-12)
        assert ex09_u(
            OptionType.PUT, x, 0.0, strike, sigma, r, q
        ) == pytest.approx(ex09_put_u0(x, strike, c.a), abs=1e-12)
    lib = heat_transform(sigma, r, q)
    assert lib.a == pytest.approx(-c.a, abs=1e-14)
    assert lib.b == pytest.approx(-c.b, abs=1e-12)


@pytest.mark.parametrize("q", [0.0, 0.03])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_ex09_heat_equation(q, option_type):
    for x in (-0.3, 0.0, 0.2):
        for tau in (0.01, 0.05):
            assert ex09_heat_residual(
                option_type, x, tau, 50.0, 0.3, 0.05, q
            ) == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize("q", [0.0, 0.01, 0.04])
@pytest.mark.parametrize("sigma", [0.2, 0.4])
def test_ex10_system(q, sigma):
    r = 0.05
    s = ex10_solve(sigma, r, q)
    first, second = ex10_residuals(s.a, s.b, sigma, r, q)
    assert first == pytest.approx(0.0, abs=1e-13) and second == pytest.approx(
        0.0, abs=1e-12
    )
    c = ex09_coefficients(sigma, r, q)
    assert s.a == pytest.approx(c.a, abs=1e-14) and s.b == pytest.approx(
        c.b, abs=1e-12
    )
    _, printed = ex10_residuals(s.a, ex10_printed_b(sigma, r, q), sigma, r, q)
    assert printed == pytest.approx(-2 * q / sigma**2, abs=1e-12)


def test_ex11_boundary_condition():
    spot, strike, barrier, maturity, sigma, r, rebate = (
        40.0,
        40.0,
        35.0,
        0.5,
        0.3,
        0.05,
        2.0,
    )
    c = ex09_coefficients(sigma, r)
    for tau in (0.0, 0.01, 0.0225):
        u = ex11_u_boundary(tau, barrier, strike, rebate, sigma, r)
        assert math.exp(
            -c.a * math.log(barrier / strike) - c.b * tau
        ) * u == pytest.approx(rebate, abs=1e-13)
    assert ex11_rebate_value(barrier, barrier, maturity, sigma, r) == 1.0
    for q in (0.0, 0.02):
        for paid in (0.0, rebate):
            cf = ex11_down_and_out_call_with_rebate(
                spot, strike, barrier, paid, maturity, sigma, r, q
            )
            assert ex11_heat_fd(
                spot, strike, barrier, paid, maturity, sigma, r, q, 400
            ) == pytest.approx(cf, abs=2e-4)
    assert ex11_down_and_out_call_with_rebate(
        spot, strike, barrier, 0.0, maturity, sigma, r
    ) == pytest.approx(3.398883, abs=1e-6)
    assert ex11_down_and_out_call_with_rebate(
        spot, strike, barrier, rebate, maturity, sigma, r
    ) == pytest.approx(4.439140, abs=1e-6)


def test_ex12_geometric_asian():
    m = AsianMarket()

    def price_f(y, t):
        return ex12_geometric_asian_call(y, t, m)

    for y in (math.log(45.0), math.log(50.0), math.log(56.0)):
        for t in (0.2, 0.6):
            assert ex12_f_residual(
                price_f, y, t, m.maturity, m.sigma, m.r
            ) == pytest.approx(0.0, abs=1e-5)

    def value_fn(spot, integral, t):
        return price_f(ex12_y(spot, integral, t, m.maturity), t)

    for spot in (44.0, 50.0, 57.0):
        t = 0.3
        assert ex12_v_residual(
            value_fn, spot, t * math.log(48.0), t, m.sigma, m.r, 1e-5
        ) == pytest.approx(0.0, abs=2e-5)
    bs = bs_call(
        50.0,
        m.strike,
        m.maturity,
        m.sigma / math.sqrt(3),
        m.r,
        0.5 * (m.r + m.sigma**2 / 6),
    )
    assert value_fn(50.0, 0.0, 0.0) == pytest.approx(bs, abs=1e-12)


def test_ex12_operator_identity():
    maturity, sigma, r = 1.0, 0.25, 0.03

    def price_f(y, t):
        return math.sin(y) * math.exp(t) + y**3 / 10

    def value_fn(spot, integral, t):
        return price_f(ex12_y(spot, integral, t, maturity), t)

    for spot in (30.0, 60.0):
        for t in (0.25, 0.7):
            integral = 0.4
            assert ex12_v_residual(
                value_fn, spot, integral, t, sigma, r
            ) == pytest.approx(
                ex12_f_residual(
                    price_f,
                    ex12_y(spot, integral, t, maturity),
                    t,
                    maturity,
                    sigma,
                    r,
                ),
                abs=1e-5,
            )


def test_ex13_arithmetic_asian():
    maturity, sigma, r = 1.0, 0.3, 0.05

    def linear_h(ratio, t):
        return ex13_h_linear(ratio, t, maturity, r)

    for ratio in (0.0, 0.3, 1.2):
        assert ex13_h_residual(linear_h, ratio, 0.4, sigma, r) == pytest.approx(
            0.0, abs=1e-7
        )
    for spot in (40.0, 55.0):
        assert ex13_v_residual(
            lambda s, integral, t: s * linear_h(integral / s, t),
            spot,
            12.0,
            0.4,
            sigma,
            r,
        ) == pytest.approx(0.0, abs=1e-5 * spot)

    def generic_h(ratio, t):
        return math.exp(-ratio * ratio) * math.cos(t) + ratio**3

    for spot in (20.0, 45.0):
        for integral in (5.0, 30.0):
            lhs = ex13_v_residual(
                lambda s, i, t: s * generic_h(i / s, t),
                spot,
                integral,
                0.5,
                sigma,
                r,
            )
            assert lhs == pytest.approx(
                spot
                * ex13_h_residual(generic_h, integral / spot, 0.5, sigma, r),
                abs=1e-4 * spot * (1 + (integral / spot) ** 3),
            )
