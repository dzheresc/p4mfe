import math

import pytest

from p4mfe.ch01_calculus_options.options import OptionType, put_bounds, put_from_call
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_greeks, bs_put
from p4mfe.ch03_probability_black_scholes.exercises.ex22_24_maturity_bounds import (
    EX22_MATURITIES,
    ex22_atm_gamma,
    ex23_atm_vega,
    ex23_vega_maturity_slope,
    ex24_call_bounds,
    ex24_put_bounds,
)
from p4mfe.ch03_probability_black_scholes.exercises.ex25_29_hedging import (
    ex25_options_only,
    ex26_shares,
    ex27,
    ex28,
    ex29_hedge,
)
from p4mfe.ch03_probability_black_scholes.hedging import delta_gamma_hedge


def test_ex22_23():
    for maturity, g, v in zip(
        EX22_MATURITIES,
        (0.130022, 0.052530, 0.025296),
        (4.069779, 9.945546, 19.723967),
    ):
        assert ex22_atm_gamma(maturity) == pytest.approx(g, abs=1e-6)
        assert ex23_atm_vega(maturity) == pytest.approx(v, abs=1e-6)
    for tau in (0.1, 1.0, 10.0, 44.0, 60.0):
        h = 1e-5
        assert ex23_vega_maturity_slope(tau) == pytest.approx(
            (ex23_atm_vega(tau + h) - ex23_atm_vega(tau - h)) / (2 * h),
            abs=1e-6,
        )
    assert ex23_vega_maturity_slope(44.0) > 0 > ex23_vega_maturity_slope(45.0)
    assert bs_greeks(
        OptionType.PUT, 50, 55, 0.7, 0.3, 0.02, 0.01
    ).vega == pytest.approx(
        bs_greeks(OptionType.CALL, 50, 55, 0.7, 0.3, 0.02, 0.01).vega
    )


def test_ex24():
    strike, maturity, r = 100.0, 1.2, 0.05
    for spot in (60.0, 100.0, 140.0):
        for sigma in (0.05, 0.3, 1.5):
            for q in (0.0, 0.03):
                cb, pb = ex24_call_bounds(
                    spot, strike, maturity, r, q
                ), ex24_put_bounds(spot, strike, maturity, r, q)
                call, put = bs_call(
                    spot, strike, maturity, sigma, r, q
                ), bs_put(spot, strike, maturity, sigma, r, q)
                assert cb.lower <= call + 1e-12 and call <= cb.upper + 1e-12
                assert pb.lower <= put + 1e-12 and put <= pb.upper + 1e-12
                assert put_from_call(
                    0.0, spot, strike, maturity, r, q
                ) == pytest.approx(pb.lower, abs=1e-12)
                assert put_from_call(
                    cb.upper, spot, strike, maturity, r, q
                ) == pytest.approx(pb.upper, abs=1e-12)
                assert max(pb.lower, 0.0) == pytest.approx(
                    put_bounds(spot, strike, maturity, r, q).lower, abs=1e-12
                )


def test_ex25():
    h = ex25_options_only(5000, -200, 0.4, 0.05, -0.5, 0.07)
    assert h.calls == pytest.approx(
        -4716.98, abs=0.01
    ) and h.puts == pytest.approx(6226.42, abs=0.01)
    assert 5000 + 0.4 * h.calls - 0.5 * h.puts == pytest.approx(0, abs=1e-9)
    assert -200 + 0.05 * h.calls + 0.07 * h.puts == pytest.approx(0, abs=1e-9)
    c = delta_gamma_hedge(5000, -200, 0.4, 0.05)
    assert (c.hedge_options, c.shares) == pytest.approx((4000, -6600))


def test_ex26_27_28():
    assert ex26_shares() == pytest.approx(-653.5007, abs=1e-4)
    e = ex27()
    assert e.premium == pytest.approx(9634.8766, abs=1e-4)
    assert e.shares == pytest.approx(-588.5891, abs=1e-4)
    assert e.position_value == pytest.approx(-49228.6098, abs=1e-4)
    assert e.portfolio_value == pytest.approx(5.1926, abs=1e-4)
    assert e.unhedged_loss == pytest.approx(1181.7533, abs=1e-4)
    f = ex28()
    assert f.value0 == pytest.approx(22927.3505, abs=1e-4)
    assert f.port_delta0 == pytest.approx(-402.7713, abs=1e-4)
    assert f.cash1 == pytest.approx(1944.5731, abs=1e-4)
    assert f.value1 == pytest.approx(23399.3380, abs=1e-4)
    assert f.shares1 - f.shares2 == pytest.approx(291.6371, abs=1e-4)


def test_ex29():
    h = ex29_hedge()
    assert (h.calls, h.puts, h.shares) == pytest.approx((-670, -110, -254))
    assert 100 + 0.1 * h.calls + 0.3 * h.puts == pytest.approx(0, abs=1e-9)
    assert 89 + 0.1 * h.calls + 0.2 * h.puts == pytest.approx(0, abs=1e-9)
    assert math.isclose(
        300 + 0.2 * h.calls - 0.8 * h.puts + h.shares, 0, abs_tol=1e-9
    )
