import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.calculus import derivative, second_derivative
from p4mfe.ch01_calculus_options.options import OptionType, payoff
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_greeks, bs_price, bs_put
from p4mfe.ch07_finite_differences_bs_pde.bs_pde import bs_pde_residual_fd
from p4mfe.ch08_multivariable_barriers.barrier import (
    barrier_image,
    down_and_in_call,
    down_and_out_call,
    up_and_in_put,
    up_and_out_put,
)
from p4mfe.ch08_multivariable_barriers.box_muller import box_muller, box_muller_inverse, box_muller_samples
from p4mfe.ch08_multivariable_barriers.double_integrals import jacobian_determinant
from p4mfe.ch08_multivariable_barriers.early_exercise import (
    ExerciseStyle,
    binomial_price,
    call_exercise_loss,
    call_exercise_loss_bound,
    early_exercise_premium,
)
from p4mfe.ch08_multivariable_barriers.heat_equation import (
    bs_from_heat,
    bs_heat_fd,
    bs_price_via_heat,
    european_heat_fd,
    heat_from_bs,
    heat_transform,
    solve_heat_explicit,
    to_heat_variables,
)
from p4mfe.ch08_multivariable_barriers.theta import theta_by_integral, theta_from_maturity
from p4mfe.normal import norm_cdf, norm_pdf


# 8.5 -----------------------------------------------------------------------
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_theta(option_type):
    spot, strike, maturity, sigma, r, q = 50, 52, 0.75, 0.3, 0.04, 0.01
    exact = bs_greeks(option_type, spot, strike, maturity, sigma, r, q).theta
    assert theta_from_maturity(
        lambda tau: bs_price(option_type, spot, strike, tau, sigma, r, q),
        maturity,
    ) == pytest.approx(exact, abs=1e-7)
    assert theta_by_integral(
        lambda s: payoff(option_type, s, strike), spot, maturity, sigma, r, q
    ) == pytest.approx(exact, abs=1e-4)


def test_theta_power_payoff():
    spot, maturity, sigma, r, q = 50, 0.75, 0.3, 0.04, 0.01
    value = spot * spot * math.exp((r - 2 * q + sigma * sigma) * maturity)
    assert theta_by_integral(
        lambda s: s * s, spot, maturity, sigma, r, q
    ) == pytest.approx(-(r - 2 * q + sigma * sigma) * value, rel=1e-4)


# 8.7 -----------------------------------------------------------------------
def test_box_muller_transform():
    z1, z2 = box_muller(math.exp(-2.0), 0.125)
    assert z1 == pytest.approx(math.sqrt(2)) and z2 == pytest.approx(
        math.sqrt(2)
    )
    u1, u2 = box_muller_inverse(-0.7, 1.3)
    assert box_muller(u1, u2) == pytest.approx((-0.7, 1.3), abs=1e-14)
    for z in ((0.3, -1.2), (1.5, 0.4), (-0.2, -0.9)):
        assert jacobian_determinant(box_muller_inverse, *z) == pytest.approx(
            norm_pdf(z[0]) * norm_pdf(z[1]), abs=1e-9
        )


def test_box_muller_samples():
    x = box_muller_samples(200_000, 11)
    assert len(box_muller_samples(5)) == 5
    assert x.mean() == pytest.approx(0, abs=0.01)
    assert (x * x).mean() == pytest.approx(1, abs=0.01)
    assert (x**4).mean() == pytest.approx(3, abs=0.05)
    assert (x < -1).mean() == pytest.approx(norm_cdf(-1.0), abs=0.003)
    assert (x[0::2] * x[1::2]).mean() == pytest.approx(0, abs=0.01)


# 8.8 -----------------------------------------------------------------------
def test_heat_transform():
    strike, maturity, sigma, r, q = 40, 0.5, 0.25, 0.05, 0.02
    c = heat_transform(sigma, r, q)

    def u(x, tau):
        return heat_from_bs(
            bs_call(
                strike * math.exp(x), strike, 2 * tau / sigma**2, sigma, r, q
            ),
            x,
            tau,
            strike,
            c,
        )

    for x in (-0.2, 0.0, 0.3):
        tau = 0.01
        ut = derivative(lambda s: u(x, s), tau)
        uxx = second_derivative(lambda y: u(y, tau), x)
        assert ut == pytest.approx(uxx, abs=1e-6 * max(1, abs(ut)))
    x, tau = to_heat_variables(44, strike, maturity, sigma)
    assert bs_from_heat(
        heat_from_bs(3.0, x, tau, strike, c), x, tau, strike, c
    ) == pytest.approx(3.0)
    for option_type in OptionType:
        for spot in (35.0, 40.0, 46.0):
            assert bs_price_via_heat(
                lambda s: payoff(option_type, s, strike),
                spot,
                strike,
                maturity,
                sigma,
                r,
                q,
            ) == pytest.approx(
                bs_price(option_type, spot, strike, maturity, sigma, r, q),
                abs=1e-7,
            )


def test_heat_fd():
    strike, maturity, sigma, r, q = 40, 0.5, 0.25, 0.05, 0.02
    for option_type in OptionType:
        for spot in (35.0, 40.0, 46.0):
            assert european_heat_fd(
                option_type, spot, strike, maturity, sigma, r, q
            ) == pytest.approx(
                bs_price(option_type, spot, strike, maturity, sigma, r, q),
                abs=2e-3,
            )
    exact = bs_call(46, strike, maturity, sigma, r, q)
    e1 = abs(
        european_heat_fd(
            OptionType.CALL, 46, strike, maturity, sigma, r, q, 100
        )
        - exact
    )
    e2 = abs(
        european_heat_fd(
            OptionType.CALL, 46, strike, maturity, sigma, r, q, 800
        )
        - exact
    )
    assert e1 / e2 > 20
    g = solve_heat_explicit(
        lambda x: np.maximum(x, 0.0),
        lambda t: 0.0,
        lambda t: 1.0,
        -1,
        1,
        0.1,
        40,
        20,
    )
    assert g.alpha > 0.5 and abs(g.u[20]) > 1e3


# 8.9 -----------------------------------------------------------------------
@pytest.mark.parametrize("barrier", [80.0, 95.0, 110.0])
def test_down_and_out_call(barrier):
    strike, maturity, sigma, r, q = 100, 1, 0.3, 0.05, 0.02
    for spot in (barrier + 1, 105.0, 130.0):
        if spot <= barrier:
            continue
        out = down_and_out_call(spot, strike, barrier, maturity, sigma, r, q)
        vanilla = bs_call(spot, strike, maturity, sigma, r, q)
        assert 0 < out < vanilla
        assert out + down_and_in_call(
            spot, strike, barrier, maturity, sigma, r, q
        ) == pytest.approx(vanilla, abs=1e-12)
        assert bs_pde_residual_fd(
            lambda s, t: down_and_out_call(
                s, strike, barrier, maturity - t, sigma, r, q
            ),
            spot,
            0.0,
            sigma,
            r,
            q,
        ) == pytest.approx(0, abs=1e-5)
        spot_max = strike * math.exp(6 * sigma)
        fd = bs_heat_fd(
            lambda s: np.where(s > barrier, np.maximum(s - strike, 0.0), 0.0),
            lambda th: 0.0,
            lambda th: spot_max * math.exp(-q * th)
            - strike * math.exp(-r * th),
            spot,
            strike,
            maturity,
            sigma,
            r,
            q,
            barrier,
            spot_max,
            600,
        )
        assert fd == pytest.approx(out, abs=5e-3)
    assert (
        down_and_out_call(barrier, strike, barrier, maturity, sigma, r, q) == 0
    )
    assert down_and_out_call(
        barrier * (1 + 1e-9), strike, barrier, maturity, sigma, r, q
    ) == pytest.approx(0, abs=1e-6)


def test_up_and_out_put_and_image():
    strike, maturity, sigma, r, q = 100, 0.5, 0.25, 0.03, 0.0
    for barrier in (90.0, 105.0, 130.0):
        spot = barrier - 5
        out = up_and_out_put(spot, strike, barrier, maturity, sigma, r, q)
        assert out > 0
        assert out + up_and_in_put(
            spot, strike, barrier, maturity, sigma, r, q
        ) == pytest.approx(
            bs_put(spot, strike, maturity, sigma, r, q), abs=1e-12
        )
        assert up_and_out_put(
            barrier * (1 - 1e-9), strike, barrier, maturity, sigma, r, q
        ) == pytest.approx(0, abs=1e-6)
    barrier = 90

    def call(s, t):
        return bs_call(s, strike, maturity - t, sigma, r, 0.01)

    def img(s, t):
        return barrier_image(lambda x: call(x, t), s, barrier, sigma, r, 0.01)

    assert bs_pde_residual_fd(img, 110, 0.1, sigma, r, 0.01) == pytest.approx(
        0, abs=1e-6
    )
    assert img(barrier, 0.0) == pytest.approx(call(barrier, 0.0), abs=1e-12)


# 8.10 ----------------------------------------------------------------------
def test_early_exercise():
    spot, strike, maturity, sigma, r = 100, 95, 1, 0.2, 0.05
    assert early_exercise_premium(
        OptionType.CALL, spot, strike, maturity, sigma, r, 0.0
    ) == pytest.approx(0, abs=1e-12)
    assert binomial_price(
        OptionType.CALL,
        spot,
        strike,
        maturity,
        sigma,
        r,
        0,
        1000,
        ExerciseStyle.EUROPEAN,
    ) == pytest.approx(bs_call(spot, strike, maturity, sigma, r), abs=2e-2)
    for tau in (0.1, 0.5, 2.0):
        for s in (80.0, 100.0, 200.0):
            assert (
                call_exercise_loss(s, strike, tau, sigma, r)
                >= call_exercise_loss_bound(strike, tau, r) - 1e-12
            )
    assert (
        early_exercise_premium(
            OptionType.CALL, 150, strike, maturity, sigma, r, 0.08
        )
        > 0.1
    )
    assert (
        early_exercise_premium(
            OptionType.PUT, 80, strike, maturity, sigma, r, 0.0
        )
        > 0.1
    )
    with pytest.raises(ValueError):
        binomial_price(
            OptionType.CALL,
            spot,
            strike,
            maturity,
            0.01,
            0.5,
            0,
            2,
            ExerciseStyle.EUROPEAN,
        )
