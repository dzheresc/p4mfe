import math

import numpy as np
import pytest

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call, bs_d1_d2, bs_greeks
from p4mfe.ch08_multivariable_barriers.barrier import down_and_in_call, down_and_out_call
from p4mfe.ch08_multivariable_barriers.early_exercise import ExerciseStyle, binomial_price, early_exercise_premium
from p4mfe.ch08_multivariable_barriers.exercises.ex14_itm_puts import EX14_R, EX14_S, EX14_SIGMA, ex14_table
from p4mfe.ch08_multivariable_barriers.exercises.ex15_time_value import Ex15Market, ex15_derivative, ex15_time_value
from p4mfe.ch08_multivariable_barriers.exercises.ex16_17_down_and_out_call import (
    Ex16Market,
    ex16_formula_846,
    ex16_value,
    ex17_bracket,
    ex17_n_d2_over_b,
)
from p4mfe.ch08_multivariable_barriers.exercises.ex18_doc_greeks import ex18_greeks, ex18_greeks_fd
from p4mfe.ch08_multivariable_barriers.exercises.ex19_20_early_exercise import (
    CallMarket,
    ex19_lower_bound_gap,
    ex19_time_value,
    ex20_crossover,
    ex20_threshold,
)
from p4mfe.ch08_multivariable_barriers.exercises.ex21_implied_vol_skew import (
    SmileMarket,
    ex21_dsigma_dk,
    ex21_price,
    ex21_slope_from_prices,
    ex21_smile,
    ex21_smile_slope,
)
from p4mfe.normal import norm_cdf


def test_ex14_table():
    values = [5.819577, 5.340299, 8.032528, 7.823400, 10.486171, 10.547645]
    larger = [False, False, False, True, True, True]
    for row, v, big in zip(ex14_table(), values, larger):
        assert row.value == pytest.approx(v, abs=1e-6)
        assert row.intrinsic_larger == big
        if big:
            am = binomial_price(
                OptionType.PUT,
                EX14_S,
                row.strike,
                row.maturity,
                EX14_SIGMA,
                EX14_R,
                0.0,
                500,
                ExerciseStyle.AMERICAN,
            )
            assert am > row.value and am >= row.intrinsic - 1e-12


@pytest.mark.parametrize(
    "m",
    [
        Ex15Market(),
        Ex15Market(100.0, 2.0, 0.2, 0.03, 0.02),
        Ex15Market(50.0, 0.1, 0.5, 0.0, 0.0),
    ],
)
def test_ex15_time_value_largest_at_the_money(m):
    at_strike = ex15_time_value(m.strike, m)
    assert at_strike == pytest.approx(
        bs_call(m.strike, m.strike, m.maturity, m.sigma, m.r, m.q), abs=1e-14
    )
    for spot in np.arange(0.2 * m.strike, 5 * m.strike, 0.005 * m.strike):
        if abs(spot - m.strike) < 1e-9:
            continue
        assert ex15_time_value(spot, m) < at_strike
        assert (ex15_derivative(spot, m) > 0) == (spot < m.strike)


def test_ex16_down_and_out_call():
    m = Ex16Market()
    assert ex16_formula_846(m) == pytest.approx(3.398883, abs=1e-6)
    assert ex16_value(m) == pytest.approx(ex16_formula_846(m), abs=1e-12)
    assert down_and_in_call(
        m.spot, m.strike, m.barrier, m.maturity, m.sigma, m.r
    ) == pytest.approx(0.455067, abs=1e-6)


def test_ex17_barrier_to_zero():
    spot, strike, maturity, sigma = 40.0, 40.0, 2.0, 0.6
    for barrier in (5.0, 20.0, 35.0):
        assert down_and_out_call(
            spot, strike, barrier, maturity, sigma, 0.0
        ) == pytest.approx(
            bs_call(spot, strike, maturity, sigma, 0.0)
            - ex17_bracket(spot, strike, barrier, maturity, sigma),
            abs=1e-12,
        )
    barriers = 35.0 * 0.7 ** np.arange(30)
    brackets = [
        abs(ex17_bracket(spot, strike, barrier, maturity, sigma))
        for barrier in barriers
    ]
    ratios = [
        ex17_n_d2_over_b(spot, strike, barrier, maturity, sigma)
        for barrier in barriers
    ]
    assert np.all(np.diff(brackets) <= 0) and np.all(np.diff(ratios) <= 0)
    assert brackets[-1] < 1e-12 and ratios[-1] < 1e-12


def test_ex18_doc_greeks():
    g = ex18_greeks(40.0, 40.0, 35.0, 0.5, 0.3, 0.05)
    assert g.delta == pytest.approx(0.703253, abs=1e-6)
    assert g.gamma == pytest.approx(0.018758, abs=1e-6)
    for q in (0.0, 0.03):
        for spot in (35.5, 36.0, 38.0, 40.0, 45.0, 60.0):
            a, f = ex18_greeks(
                spot, 40.0, 35.0, 0.5, 0.3, 0.05, q
            ), ex18_greeks_fd(spot, 40.0, 35.0, 0.5, 0.3, 0.05, q)
            assert a.delta == pytest.approx(f.delta, abs=1e-6)
            assert a.gamma == pytest.approx(f.gamma, abs=1e-5)
    assert ex18_greeks(36.0, 40.0, 35.0, 0.5, 0.3, 0.05).gamma < 0


def test_ex19_call_above_intrinsic():
    for maturity in (0.05, 1.0, 5.0):
        for sigma in (0.05, 0.3, 0.8):
            m = CallMarket(40.0, maturity, sigma, 0.05, 0.0)
            spot = 1.0
            while spot < 400.0:
                tv = ex19_time_value(spot, m)
                assert tv >= 0.0  # underflows to 0 far out of the money
                if spot > 0.8 * m.strike:
                    assert tv > 0.0
                if spot > m.strike:
                    assert tv >= ex19_lower_bound_gap(m) * (1 - 1e-9)
                spot *= 1.1
    for spot in (30.0, 40.0, 60.0):
        assert early_exercise_premium(
            OptionType.CALL, spot, 40.0, 1.0, 0.3, 0.05, 0.0, 400
        ) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("q", [0.05, 0.01, 0.001])
def test_ex20_dividend_call_below_intrinsic(q):
    m = CallMarket(40.0, 1.0, 0.3, 0.05, q)
    spot_threshold, spot_crossover = ex20_threshold(m), ex20_crossover(m)
    assert m.strike < spot_crossover < spot_threshold
    assert ex19_time_value(spot_threshold, m) < 0
    assert (
        ex19_time_value(1.01 * spot_crossover, m)
        < 0
        < ex19_time_value(0.99 * spot_crossover, m)
    )


def test_ex20_values():
    assert ex20_crossover(
        CallMarket(40.0, 1.0, 0.3, 0.05, 0.01)
    ) == pytest.approx(196.0594, abs=1e-3)
    assert ex20_crossover(
        CallMarket(40.0, 1.0, 0.3, 0.05, 0.05)
    ) == pytest.approx(56.7112, abs=1e-3)


def test_ex21_implied_vol_slope():
    m = SmileMarket()
    for strike in (70.0, 90.0, 100.0, 115.0, 140.0):
        s = ex21_smile(strike, m)
        _, d2 = bs_d1_d2(m.spot, strike, m.maturity, s, m.r, m.q)
        vega = bs_greeks(
            OptionType.CALL, m.spot, strike, m.maturity, s, m.r, m.q
        ).vega
        d_call = -math.exp(-m.r * m.maturity) * norm_cdf(
            d2
        ) + vega * ex21_smile_slope(strike, m)
        assert ex21_dsigma_dk(strike, s, d_call, m) == pytest.approx(
            ex21_smile_slope(strike, m), abs=1e-12
        )
        assert ex21_slope_from_prices(
            lambda k: ex21_price(k, m), strike, m
        ) == pytest.approx(ex21_smile_slope(strike, m), abs=1e-7)
    flat = lambda k: bs_call(m.spot, k, m.maturity, 0.25, m.r, m.q)  # noqa: E731
    assert ex21_slope_from_prices(flat, 100.0, m) == pytest.approx(
        0.0, abs=1e-7
    )
