import math

import pytest

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_d1_d2, bs_greeks, bs_put
from p4mfe.ch03_probability_black_scholes.exercises.ex12_15_black_scholes import (
    ex12_call,
    ex12_put,
    ex13_call_zero_strike,
    ex13_put_zero_strike,
    ex14_rho_put_from_parity,
    ex15_aon_call,
    ex15_aon_delta,
)
from p4mfe.ch03_probability_black_scholes.exercises.ex16_21_greeks_strike import (
    ex16_vanna,
    ex16_volga,
    ex17_atm_call_minus_put,
    ex18_atm_q_eq_r_bracket,
    ex18_theta_call,
    ex18_theta_sign_change,
    ex19_strike_derivatives,
    ex21_min_straddle_strike,
    ex21_straddle,
)

CASES = [(50, 45, 0.5, 0.2, 0.06, 0.02), (100, 110, 1.5, 0.35, 0.03, 0.0), (40, 38, 0.25, 0.15, 0.01, 0.04), (75, 75, 3.0, 0.25, 0.05, 0.05)]


def test_ex12():
    d1, d2 = bs_d1_d2(50, 45, 0.5, 0.2, 0.06, 0.02)
    assert d1 == pytest.approx(0.957143, abs=1e-6) and d2 == pytest.approx(0.815722, abs=1e-6)
    assert ex12_call() == pytest.approx(6.508365, abs=1e-6)
    assert ex12_put() == pytest.approx(0.675922, abs=1e-6)
    assert ex12_call() - ex12_put() == pytest.approx(50 * math.exp(-0.01) - 45 * math.exp(-0.03), abs=1e-12)


def test_ex13():
    assert ex13_put_zero_strike() == 0
    assert bs_call(80, 1e-8, 2.0, 0.3, 0.05, 0.03) == pytest.approx(ex13_call_zero_strike(80, 2.0, 0.03), abs=1e-7)
    assert bs_put(80, 1e-8, 2.0, 0.3, 0.05, 0.03) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("S,K,T,sigma,r,q", CASES)
def test_ex14_15(S, K, T, sigma, r, q):
    rho = bs_greeks(OptionType.PUT, S, K, T, sigma, r, q).rho
    assert ex14_rho_put_from_parity(S, K, T, sigma, r, q) == pytest.approx(rho, abs=1e-10)
    h = 1e-4
    fd = (ex15_aon_call(S + h, K, T, 0.3, r, q) - ex15_aon_call(S - h, K, T, 0.3, r, q)) / (2 * h)
    assert ex15_aon_delta(S, K, T, 0.3, r, q) == pytest.approx(fd, abs=1e-7)


@pytest.mark.parametrize("S,K,T,sigma,r,q", CASES)
def test_ex16(S, K, T, sigma, r, q):
    h = 1e-4
    for t in OptionType:
        vega = lambda S_, s_: bs_greeks(t, S_, K, T, s_, r, q).vega  # noqa: E731
        assert ex16_volga(S, K, T, sigma, r, q) == pytest.approx((vega(S, sigma + h) - vega(S, sigma - h)) / (2 * h), abs=1e-5)
        assert ex16_vanna(S, K, T, sigma, r, q) == pytest.approx((vega(S + h, sigma) - vega(S - h, sigma)) / (2 * h), abs=1e-6)


def test_ex17():
    for r in (0.01, 0.04, 0.08):
        for q in (0.0, 0.04, 0.1):
            diff = ex17_atm_call_minus_put(60, 1.5, 0.25, r, q)
            assert diff == pytest.approx(60 * (math.exp(-q * 1.5) - math.exp(-r * 1.5)), abs=1e-11)
            assert (diff > 0) == (q < r) or q == r


def test_ex18():
    for S in (20, 50, 80):
        for T in (0.01, 0.5, 5, 50):
            for r in (0, 0.03, 0.1):
                assert ex18_theta_call(S, 50, T, 0.3, r) < 0
    assert ex18_theta_sign_change(0.2, 0.05) == pytest.approx(9.6814, abs=1e-4)
    assert ex18_theta_call(100, 100, 1, 0.2, 0.05, 0.05) < 0 < ex18_theta_call(100, 100, 10, 0.2, 0.05, 0.05)
    for T in (0.5, 5, 20):
        assert ex18_theta_call(100, 100, T, 0.2, 0.05, 0.05) == pytest.approx(
            100 * math.exp(-0.05 * T) * ex18_atm_q_eq_r_bracket(T, 0.2, 0.05), abs=1e-11)


@pytest.mark.parametrize("S,K,T,sigma,r,q", CASES)
def test_ex19_20_21(S, K, T, sigma, r, q):
    h = 1e-3
    C = lambda k: bs_call(S, k, T, sigma, r, q)  # noqa: E731
    P = lambda k: bs_put(S, k, T, sigma, r, q)  # noqa: E731
    d = ex19_strike_derivatives(S, K, T, sigma, r, q)
    assert d.dC_dK == pytest.approx((C(K + h) - C(K - h)) / (2 * h), abs=1e-8)
    assert d.dP_dK == pytest.approx((P(K + h) - P(K - h)) / (2 * h), abs=1e-8)
    assert d.d2C_dK2 == pytest.approx((C(K + h) - 2 * C(K) + C(K - h)) / h**2, abs=1e-5)
    assert d.d2P_dK2 == pytest.approx((P(K + h) - 2 * P(K) + P(K - h)) / h**2, abs=1e-5)
    assert d.d2C_dK2 > 0
    Ks = ex21_min_straddle_strike(S, T, sigma, r, q)
    assert bs_d1_d2(S, Ks, T, sigma, r, q)[1] == pytest.approx(0, abs=1e-12)
    f = ex21_straddle(S, Ks, T, sigma, r, q)
    for dK in (-1, -0.1, 0.1, 1):
        assert ex21_straddle(S, Ks + dK, T, sigma, r, q) > f
