import math

import pytest

from p4mfe.ch01_calculus_options.options import OptionType, call_bounds
from p4mfe.ch02_integration_bonds.bonds import bond_cash_flows, bond_price, bond_price_discrete, bond_yield
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_d1_d2, bs_greeks
from p4mfe.ch03_probability_black_scholes.implied_vol import implied_vol_newton as ch03_implied_vol_newton
from p4mfe.ch05_newton_bootstrapping.bond_yield import bond_yield_bisection
from p4mfe.ch05_newton_bootstrapping.exercises.ex01_bond_yield import ex01_solve
from p4mfe.ch05_newton_bootstrapping.exercises.ex02_zero_curve_bond import ex02_cash_flows, ex02_solve, ex02_zero_rate
from p4mfe.ch05_newton_bootstrapping.exercises.ex03_implied_vol_limits import (
    Ex03Market,
    ex03_d_limit_zero,
    ex03_f,
    ex03_has_positive_root,
    ex03_limit_infinity,
    ex03_limit_zero,
    ex03_price_range,
)
from p4mfe.ch05_newton_bootstrapping.exercises.ex04_implied_vol_methods import (
    Ex04Market,
    ex04_bisection,
    ex04_newton,
    ex04_relative_error,
    ex04_secant,
    ex04_sigma_approx,
)
from p4mfe.ch05_newton_bootstrapping.implied_vol import implied_vol_newton


def test_ex01():
    r = ex01_solve()
    assert r.yield_ == pytest.approx(0.033401, abs=5e-7)
    assert r.duration == pytest.approx(4.642735, abs=5e-7)
    assert r.convexity == pytest.approx(22.573118, abs=5e-7)
    cf = bond_cash_flows(5.0, 0.03375, 2)
    price = 100 + 1 / 32
    assert bond_price(cf, r.yield_) == pytest.approx(price, abs=1e-10)
    h = 1e-4
    price_up, price_down = bond_price(cf, r.yield_ + h), bond_price(
        cf, r.yield_ - h
    )
    assert -(price_up - price_down) / (2 * h) / price == pytest.approx(
        r.duration, abs=1e-6
    )
    assert (
        price_up - 2 * price + price_down
    ) / h**2 / price == pytest.approx(r.convexity, abs=1e-4)
    assert bond_yield_bisection(cf, price).x == pytest.approx(
        r.yield_, abs=1e-10
    )
    assert bond_yield(cf, price) == pytest.approx(r.yield_, abs=1e-10)


def test_ex02():
    r = ex02_solve()
    assert r.price == pytest.approx(104.173911, abs=5e-7)
    assert r.yield_ == pytest.approx(0.056792, abs=5e-7)
    assert r.duration == pytest.approx(1.890108, abs=5e-7)
    assert r.convexity == pytest.approx(3.689501, abs=5e-7)
    assert r.yield_semiannual == pytest.approx(0.057606, abs=5e-7)
    assert r.modified_duration_semiannual == pytest.approx(1.837191, abs=5e-7)
    price = sum(
        4 * math.exp(-ex02_zero_rate(t) * t) for t in (0.5, 1.0, 1.5)
    ) + 104 * math.exp(-ex02_zero_rate(2) * 2)
    assert r.price == pytest.approx(price, abs=1e-12)
    cf = ex02_cash_flows()
    assert bond_price(cf, r.yield_) == pytest.approx(price, abs=1e-10)
    assert bond_price_discrete(cf, r.yield_semiannual, 2) == pytest.approx(
        price, abs=1e-10
    )
    h = 1e-5
    d_price = (
        bond_price_discrete(cf, r.yield_semiannual + h, 2)
        - bond_price_discrete(cf, r.yield_semiannual - h, 2)
    ) / (2 * h)
    assert -d_price / price == pytest.approx(
        r.modified_duration_semiannual, abs=1e-7
    )


@pytest.mark.parametrize("case", ["F<K", "F=K", "F>K"])
def test_ex03(case):
    spot, maturity, r, q = 50.0, 0.75, 0.05, 0.01
    forward = spot * math.exp((r - q) * maturity)
    strike = {"F<K": 60.0, "F=K": forward, "F>K": 40.0}[case]
    lo, hi = ex03_price_range(spot, strike, maturity, r, q)
    m = Ex03Market(spot, strike, maturity, r, q, 0.5 * (lo + hi))
    # (i)
    assert ex03_f(200.0, m) == pytest.approx(ex03_limit_infinity(m), abs=1e-9)
    d1, d2 = bs_d1_d2(spot, strike, maturity, 200.0, r, q)
    assert d1 > 50 and d2 < -50
    # (ii)
    assert ex03_f(1e-12, m) == pytest.approx(ex03_limit_zero(m), abs=1e-9)
    d1, d2 = bs_d1_d2(spot, strike, maturity, 1e-6, r, q)
    lim = ex03_d_limit_zero(m)
    if math.isinf(lim):
        assert d1 * lim > 0 and abs(d1) > 1e3 and abs(d2) > 1e3
    else:
        assert abs(d1) < 1e-5 and abs(d2) < 1e-5
    # (iii)
    prev = ex03_limit_zero(m)
    for k in range(1, 500):
        x = 0.01 * k
        fx = ex03_f(x, m)
        assert (
            bs_greeks(OptionType.CALL, spot, strike, maturity, x, r, q).vega > 0
        )
        assert prev <= fx < ex03_limit_infinity(m)
        prev = fx
    # (iv)
    b = call_bounds(spot, strike, maturity, r, q)
    assert (lo, hi) == pytest.approx((b.lower, b.upper), abs=1e-12)
    assert ex03_has_positive_root(m)
    sigma = implied_vol_newton(
        OptionType.CALL, m.call, spot, strike, maturity, r, q
    ).x
    assert ex03_f(sigma, m) == pytest.approx(0.0, abs=1e-8)
    for call in (hi, lo, hi + 1.0):
        assert not ex03_has_positive_root(
            Ex03Market(spot, strike, maturity, r, q, call)
        )


def test_ex04():
    b, s, n = ex04_bisection(), ex04_secant(), ex04_newton()
    assert n.x == pytest.approx(0.397049, abs=5e-7)
    assert b.x == pytest.approx(n.x, abs=1e-6)
    assert s.x == pytest.approx(n.x, abs=1e-6)
    assert n.iterations == 3 and s.iterations == 3 and b.iterations >= 20
    m = Ex04Market()
    assert bs_call(
        m.spot, m.strike, m.maturity, n.x, m.r, m.q
    ) == pytest.approx(m.call, abs=1e-9)
    assert ch03_implied_vol_newton(
        OptionType.CALL, m.call, m.spot, m.strike, m.maturity, m.r, m.q
    ).sigma == pytest.approx(n.x, abs=1e-8)
    assert ex04_sigma_approx() == pytest.approx(0.396672, abs=5e-7)
    assert ex04_relative_error() == pytest.approx(9.49e-4, abs=5e-6)
