import numpy as np
import pytest

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch02_integration_bonds.bonds import (
    BondQuote,
    ZeroCurve,
    bond_cash_flows,
    bond_price,
    bond_price_from_zero_rate,
    bond_yield,
    bootstrap_zero_rates,
)
from p4mfe.ch03_probability_black_scholes import implied_vol as ch03_iv
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_price
from p4mfe.ch05_newton_bootstrapping.bond_yield import bond_yield_bisection, bond_yield_newton, bond_yield_secant
from p4mfe.ch05_newton_bootstrapping.bootstrapping import bootstrap_next_rate, bootstrap_zero_curve, zero_rate_linear
from p4mfe.ch05_newton_bootstrapping.implied_vol import (
    implied_vol_bisection,
    implied_vol_newton,
    implied_vol_secant,
)
from p4mfe.ch05_newton_bootstrapping.solvers import Tolerances


# 5.3 -----------------------------------------------------------------------
@pytest.mark.parametrize("price", [90.0, 101.0, 105.0])
def test_bond_yield_methods_agree_with_chapter2(price):
    cf = bond_cash_flows(3.0, 0.06, 2)
    y2 = bond_yield(cf, price)
    n, s, b = (
        bond_yield_newton(cf, price),
        bond_yield_secant(cf, price),
        bond_yield_bisection(cf, price),
    )
    assert n.x == pytest.approx(y2, abs=1e-12)
    assert s.x == pytest.approx(y2, abs=1e-12)
    assert b.x == pytest.approx(y2, abs=1e-11)
    assert bond_price(cf, n.x) == pytest.approx(price, abs=1e-10)
    assert n.iterations <= 6 and n.iterations < b.iterations


@pytest.mark.parametrize("y", [-0.005, 0.02, 0.12])
def test_bond_yield_round_trip_long_bond(y):
    cf = bond_cash_flows(29.75, 0.045, 2)
    assert bond_yield_newton(cf, bond_price(cf, y)).x == pytest.approx(
        y, abs=1e-12
    )


# 5.4 -----------------------------------------------------------------------
@pytest.mark.parametrize("t", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("strike", [25.0, 30.0, 36.0])
@pytest.mark.parametrize("sigma", [0.15, 0.3, 0.8])
def test_implied_vol_recovers_sigma(t, strike, sigma):
    spot, maturity, r, q = 30.0, 0.5, 0.03, 0.01
    price = bs_price(t, spot, strike, maturity, sigma, r, q)
    assert implied_vol_newton(
        t, price, spot, strike, maturity, r, q
    ).x == pytest.approx(sigma, abs=1e-9)
    assert implied_vol_secant(
        t, price, spot, strike, maturity, r, q
    ).x == pytest.approx(sigma, abs=1e-9)
    assert implied_vol_bisection(
        t, price, spot, strike, maturity, r, q
    ).x == pytest.approx(sigma, abs=1e-8)
    assert implied_vol_newton(
        t, price, spot, strike, maturity, r, q
    ).x == pytest.approx(
        ch03_iv.implied_vol_newton(
            t, price, spot, strike, maturity, r, q
        ).sigma,
        abs=1e-9,
    )


def test_implied_vol_newton_is_monotone():
    spot, strike, maturity, r, sigma = 40, 44, 0.75, 0.04, 0.27
    res = implied_vol_newton(
        OptionType.CALL,
        bs_call(spot, strike, maturity, sigma, r),
        spot,
        strike,
        maturity,
        r,
        0.0,
        Tolerances(1e-14, 1e-14, 100),
    )
    side = res.iterates[0] - sigma
    assert all(
        (x - sigma) * side > 0 for x in res.iterates if abs(x - sigma) > 1e-14
    )


def test_implied_vol_rejects_arbitrageable_prices():
    with pytest.raises(ValueError):
        implied_vol_newton(OptionType.CALL, 31.0, 30, 30, 0.5, 0.03)
    with pytest.raises(ValueError):
        implied_vol_secant(OptionType.PUT, 0.0, 30, 30, 0.5, 0.03)


# 5.5 -----------------------------------------------------------------------
def test_linear_interpolation():
    c = ZeroCurve(np.array([0.0, 1.0, 3.0]), np.array([0.05, 0.06, 0.07]))
    assert zero_rate_linear(c, 0.5) == pytest.approx(0.055)
    assert zero_rate_linear(c, 2.0) == pytest.approx(0.065)
    assert zero_rate_linear(c, 4.0) == pytest.approx(0.07)
    assert zero_rate_linear(c, -1.0) == pytest.approx(0.05)


def test_recovers_piecewise_linear_curve():
    truth = ZeroCurve(
        np.array([0.0, 0.5, 1.0, 3.0, 5.0]),
        np.array([0.05, 0.051, 0.0525, 0.058, 0.061]),
    )
    quotes = []
    for maturity, cpn in zip(truth.times[1:], [0.0, 0.05, 0.055, 0.06]):
        cf = bond_cash_flows(maturity, cpn, 2)
        quotes.append(
            BondQuote(
                maturity,
                cpn,
                bond_price_from_zero_rate(
                    cf, lambda t: zero_rate_linear(truth, t)
                ),
            )
        )
    curve = bootstrap_zero_curve(
        quotes, 2, ZeroCurve(np.array([0.0]), np.array([0.05]))
    )
    np.testing.assert_allclose(curve.times, truth.times)
    np.testing.assert_allclose(curve.rates, truth.rates, atol=1e-12)


def test_agrees_with_chapter2_when_no_interpolation_is_needed():
    quotes = [
        BondQuote(0.5, 0.0, 97.5),
        BondQuote(1.0, 0.05, 100.0),
        BondQuote(1.5, 0.055, 100.3),
        BondQuote(2.0, 0.06, 101.0),
    ]
    np.testing.assert_allclose(
        bootstrap_zero_curve(quotes, 2).rates,
        bootstrap_zero_rates(quotes, 2).rates,
        atol=1e-12,
    )


def test_first_bond_without_curve_gets_its_yield():
    cf = bond_cash_flows(2.0, 0.04, 2)
    assert bootstrap_next_rate(None, cf, 99.0).x == pytest.approx(
        bond_yield(cf, 99.0), abs=1e-12
    )


def test_maturity_must_extend_the_curve():
    c = ZeroCurve(np.array([0.0, 1.0]), np.array([0.05, 0.06]))
    with pytest.raises(ValueError):
        bootstrap_next_rate(c, bond_cash_flows(1.0, 0.05, 2), 100.0)
