import math

import pytest

from p4mfe.ch02_integration_bonds.bonds import bond_price
from p4mfe.ch06_taylor_bonds_atm.bs_atm import atm_price_approx, atm_price_exact
from p4mfe.ch06_taylor_bonds_atm.exercises.ex10_log_product_integral import (
    EX10_EXACT,
    ex10_integrand,
    ex10_moment,
    ex10_moment_numerical,
    ex10_numerical,
    ex10_partial_sum,
)
from p4mfe.ch06_taylor_bonds_atm.exercises.ex11_crr_expansion import ex11_approx, ex11_crr, ex11_u_coefficient
from p4mfe.ch06_taylor_bonds_atm.exercises.ex12_atm_put_error import ex12_with_rates, ex12_zero_rates
from p4mfe.ch06_taylor_bonds_atm.exercises.ex13_atm_parity import (
    ex13_parity_gap,
    ex13_parity_gap_6103,
    ex13_put,
    ex13_solve,
)
from p4mfe.ch06_taylor_bonds_atm.exercises.ex14_atm_put_maturity import ex14_table
from p4mfe.ch06_taylor_bonds_atm.exercises.ex15_atm_put_dividends import ex15_table
from p4mfe.ch06_taylor_bonds_atm.exercises.ex16_bond_price_change import ex16_table
from p4mfe.ch06_taylor_bonds_atm.exercises.ex17_portfolio_immunization import (
    ex17_bond1,
    ex17_bond2,
    ex17_immunize,
    ex17_new_value,
    ex17_portfolio,
)
from p4mfe.ch06_taylor_bonds_atm.exercises.ex18_curve_shift_yield import (
    EX18_SHIFTS,
    ex18_cash_flows,
    ex18_solve,
    ex18_yield_changes,
    ex18_zero_rate,
)
from p4mfe.ch06_taylor_bonds_atm.yield_curve_shifts import shifted_curve_price


def test_ex10_log_product_integral():
    assert EX10_EXACT == pytest.approx(0.355065933151774, abs=1e-14)
    assert abs(ex10_integrand(1e-10)) < 1e-8
    assert abs(ex10_integrand(1 - 1e-10)) < 1e-8
    assert ex10_numerical() == pytest.approx(EX10_EXACT, abs=1e-8)
    for n in (0, 1, 3, 7):
        assert ex10_moment_numerical(n) == pytest.approx(ex10_moment(n), abs=1e-8)
    for N in (10, 100, 1000):
        tail = EX10_EXACT - ex10_partial_sum(N)
        assert 0 < tail < 0.5 / N**2
    for n in range(1, 51):
        assert 1 / (n * (n + 1) ** 2) == pytest.approx(1 / n - 1 / (n + 1) - 1 / (n + 1) ** 2, abs=1e-15)


@pytest.mark.parametrize("r", [0.0, 0.05])
@pytest.mark.parametrize("sigma", [0.2, 0.4])
@pytest.mark.parametrize("dt", [1e-2, 1e-3, 1e-4, 1e-5])
def test_ex11_crr_small_step(r, sigma, dt):
    c, a = ex11_crr(r, sigma, dt), ex11_approx(r, sigma, dt)
    assert c.u * c.d == pytest.approx(1.0, abs=1e-13)
    assert c.p * c.u + (1 - c.p) * c.d == pytest.approx(math.exp(r * dt), abs=1e-13)
    s = dt**1.5
    k = ex11_u_coefficient(r, sigma)
    assert (c.u - a.u) / s == pytest.approx(k, abs=0.2 * k + 1e-3)
    assert (c.d - a.d) / s == pytest.approx(-k, abs=0.2 * k + 1e-3)
    assert abs(c.p - a.p) / s < 1.0
    assert abs(c.u - a.u) < dt and abs(c.p - a.p) < dt


def test_ex12_atm_put():
    z = ex12_zero_rates()
    assert (z.approx, z.black_scholes, z.relative_error) == pytest.approx((5.984134, 5.968587, 0.002605), abs=5e-7)
    assert z.black_scholes == pytest.approx(atm_price_exact(60.0, 1.0, 0.25), abs=1e-12)
    w = ex12_with_rates()
    assert (w.approx, w.black_scholes, w.relative_error) == pytest.approx((4.814848, 4.886987, 0.014761), abs=5e-7)


def test_ex13_parity():
    for r in (0.0, 0.06, 0.1):
        for q in (0.0, 0.03):
            for T in (0.25, 1.0, 5.0):
                assert ex13_parity_gap(60.0, T, 0.25, r, q) == pytest.approx(0.0, abs=1e-12)
                if r != q:
                    assert abs(ex13_parity_gap_6103(60.0, T, 0.25, r, q)) > 1e-6
    res = ex13_solve()
    assert res.approx == pytest.approx(4.861031, abs=5e-7)
    assert res.relative_error == pytest.approx(0.005311, abs=5e-7)
    assert res.relative_error < ex12_with_rates().relative_error
    assert ex13_put(60.0, 1.0, 0.25, 0.0, 0.0) == pytest.approx(atm_price_approx(60.0, 1.0, 0.25), abs=1e-14)


def test_ex14_long_maturities():
    t = ex14_table()
    assert [r.relative_error for r in t] == pytest.approx([0.011262, 0.018784, 0.037632, 0.075497], abs=5e-7)
    assert all(r.approx > r.black_scholes for r in t)
    assert (t[3].approx, t[3].black_scholes) == pytest.approx((21.409489, 19.906602), abs=5e-7)


def test_ex15_long_maturities_with_rates():
    t = ex15_table()
    expected = [0.004199, 0.028303, 0.072609, 0.288986, 1.413676]
    assert [r.relative_error for r in t] == pytest.approx(expected, abs=5e-7)
    assert (t[0].approx, t[0].black_scholes) == pytest.approx((4.131720, 4.149140), abs=5e-7)
    assert t[4].approx < 0


def test_ex16_price_change():
    t = ex16_table()
    assert [r.duration_only for r in t] == pytest.approx([100.8485, 100.2425, 99.485, 97.97], abs=1e-10)
    assert [r.duration_convexity for r in t] == pytest.approx(
        [100.84862625, 100.24565625, 99.497625, 98.0205], abs=1e-10
    )


def test_ex17_immunization():
    assert ex17_new_value() == pytest.approx(49_505_000.0, abs=1e-6)
    h = ex17_immunize()
    assert (h.units1, h.units2) == pytest.approx((-20202.0202, -139762.4039), abs=1e-4)
    assert h.cash == pytest.approx(-12_962_962.963, abs=1e-3)
    p, b1, b2 = ex17_portfolio(), ex17_bond1(), ex17_bond2()
    assert p.dollar_duration + h.units1 * b1.dollar_duration + h.units2 * b2.dollar_duration == pytest.approx(0, abs=1e-4)
    assert p.dollar_convexity + h.units1 * b1.dollar_convexity + h.units2 * b2.dollar_convexity == pytest.approx(
        0, abs=1e-4
    )
    assert p.value + h.units1 * b1.value + h.units2 * b2.value + h.cash == pytest.approx(0, abs=1e-4)


def test_ex18_yield_under_shifts():
    r = ex18_solve()
    assert (r.price, r.yield_) == pytest.approx((106.199524, 0.042511), abs=5e-7)
    cf = ex18_cash_flows()
    assert bond_price(cf, r.yield_) == pytest.approx(r.price, abs=1e-9)
    expected = [0.00099979, 0.00199957, 0.00499893, 0.00999784, 0.01999562]
    for dr, dy, e in zip(EX18_SHIFTS, ex18_yield_changes(), expected):
        assert dy == pytest.approx(e, abs=5e-9)
        assert 0.999 * dr < dy < dr
        assert bond_price(cf, r.yield_ + dy) == pytest.approx(shifted_curve_price(cf, ex18_zero_rate, dr), abs=1e-9)
