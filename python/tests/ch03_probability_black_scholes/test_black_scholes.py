import math

import pytest

from p4mfe.ch01_calculus_options.calculus import derivative, second_derivative
from p4mfe.ch01_calculus_options.options import parity_violation
from p4mfe.ch03_probability_black_scholes.black_scholes import (
    OptionType,
    bs_call,
    bs_greeks,
    bs_pde_residual,
    bs_phi_identity_gap,
    bs_price,
    bs_put,
)

S, K, T, SIGMA, R, Q = 40.0, 42.0, 0.5, 0.27, 0.05, 0.015


def test_reference_values():
    assert bs_call(42, 40, 0.5, 0.2, 0.1) == pytest.approx(4.7594, abs=5e-5)
    assert bs_put(42, 40, 0.5, 0.2, 0.1) == pytest.approx(0.8086, abs=5e-5)
    assert bs_call(100, 100, 1, 0.2, 0.05) == pytest.approx(10.450583572185565, abs=1e-12)
    assert bs_put(100, 100, 1, 0.2, 0.05) == pytest.approx(5.573526022256971, abs=1e-12)


def test_put_call_parity_with_dividends():
    args = (40, 45, 0.75, 0.3, 0.04, 0.02)
    C, P = bs_call(*args), bs_put(*args)
    assert parity_violation(C, P, 40, 45, 0.75, 0.04, 0.02) == pytest.approx(0.0, abs=1e-12)


def test_limits_and_bad_input():
    assert bs_call(50, 40, 1, 1e-6, 0.05, 0.01) == pytest.approx(50 * math.exp(-0.01) - 40 * math.exp(-0.05), abs=1e-10)
    with pytest.raises(ValueError):
        bs_call(50, 40, 0, 0.2, 0.05)
    with pytest.raises(ValueError):
        bs_call(50, 40, 1, 0.0, 0.05)


@pytest.mark.parametrize("t", [OptionType.CALL, OptionType.PUT])
def test_greeks_match_finite_differences(t):
    g = bs_greeks(t, S, K, T, SIGMA, R, Q)
    assert g.delta == pytest.approx(derivative(lambda s: bs_price(t, s, K, T, SIGMA, R, Q), S), abs=1e-8)
    assert g.gamma == pytest.approx(second_derivative(lambda s: bs_price(t, s, K, T, SIGMA, R, Q), S), abs=1e-6)
    assert g.vega == pytest.approx(derivative(lambda v: bs_price(t, S, K, T, v, R, Q), SIGMA), abs=1e-7)
    assert g.rho == pytest.approx(derivative(lambda x: bs_price(t, S, K, T, SIGMA, x, Q), R), abs=1e-7)
    assert g.theta == pytest.approx(-derivative(lambda u: bs_price(t, S, K, u, SIGMA, R, Q), T), abs=1e-7)


@pytest.mark.parametrize("t", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("spot", [20.0, 40.0, 60.0])
def test_pde(t, spot):
    assert bs_pde_residual(t, spot, K, T, SIGMA, R, Q) == pytest.approx(0.0, abs=1e-12)


def test_call_put_relations_and_phi_identity():
    c = bs_greeks(OptionType.CALL, S, K, T, SIGMA, R, Q)
    p = bs_greeks(OptionType.PUT, S, K, T, SIGMA, R, Q)
    assert c.delta - p.delta == pytest.approx(math.exp(-Q * T), abs=1e-15)
    assert c.gamma == p.gamma and c.vega == p.vega
    assert c.rho - p.rho == pytest.approx(K * T * math.exp(-R * T), abs=1e-13)
    for spot in (10.0, 40.0, 100.0):
        assert bs_phi_identity_gap(spot, K, T, SIGMA, R, Q) == pytest.approx(0.0, abs=1e-14)
