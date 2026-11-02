import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch02_integration_bonds.integration import Rule, observed_order, simpson_rule
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_d1_d2, bs_greeks
from p4mfe.ch04_lognormal_risk_neutral.asset_model import price_from_normal
from p4mfe.ch04_lognormal_risk_neutral.exercises.ex09_10_itm_delta import (
    ex09_n_minus_d2,
    ex09_prob_itm,
    ex09_rn_prob_itm,
    ex10_atm_call_delta,
    ex10_threshold_q,
    ex10_threshold_q_approx,
)
from p4mfe.ch04_lognormal_risk_neutral.exercises.ex11_put_integration import ex11_integrand, ex11_put_bs, ex11_run
from p4mfe.ch04_lognormal_risk_neutral.exercises.ex12_13_power_payoffs import ex12_squared_call, ex13_power_call
from p4mfe.ch04_lognormal_risk_neutral.exercises.ex14_15_normal_model_inverse import (
    ex14_book_forward,
    ex14_call,
    ex14_call_forward,
    ex14_put_forward,
    ex14_put_parity,
    ex15_inverse_call,
    ex15_inverse_claim,
)

CASES = [
    (50, 45, 0.5, 0.2, 0.06, 0.0),
    (100, 110, 1.5, 0.35, 0.03, 0.0),
    (40, 38, 0.25, 0.15, 0.01, 0.04),
    (75, 75, 2.0, 0.25, 0.05, 0.02),
]


def _phi(z):
    return np.exp(-0.5 * np.asarray(z) ** 2) / math.sqrt(2 * math.pi)


def rn_price_above(g, event_strike, c):
    """e^(-rT) * Integral over {S_T > K_event} of g(S_T) phi(z) dz, Simpson on [z_K, 12]."""
    spot, strike, maturity, sigma, r, q = c
    z_lo = -bs_d1_d2(spot, event_strike, maturity, sigma, r, q)[1]
    f = lambda z: g(price_from_normal(z, spot, maturity, sigma, r, q)) * _phi(z)
    return math.exp(-r * maturity) * simpson_rule(f, z_lo, 12.0, 20000)


def test_ex09():
    assert ex09_prob_itm() == pytest.approx(0.271525, abs=1e-6)
    assert ex09_rn_prob_itm() == pytest.approx(0.304331, abs=1e-6)
    assert ex09_n_minus_d2() == pytest.approx(ex09_rn_prob_itm(), abs=1e-12)


@pytest.mark.parametrize("r", [0.0, 0.02, 0.08])
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.6])
@pytest.mark.parametrize("maturity", [0.05, 1.0, 10.0])
def test_ex10(r, sigma, maturity):
    d = ex10_atm_call_delta(r, 0.0, sigma, maturity)
    assert d > 0.5
    assert d == pytest.approx(
        bs_greeks(OptionType.CALL, 100, 100, maturity, sigma, r).delta,
        abs=1e-12,
    )
    qs = ex10_threshold_q(r, sigma, maturity)
    assert ex10_atm_call_delta(r, qs, sigma, maturity) == pytest.approx(
        0.5, abs=1e-12
    )
    assert ex10_atm_call_delta(r, 0.9 * qs, sigma, maturity) > 0.5
    assert ex10_atm_call_delta(r, 1.1 * qs + 1e-6, sigma, maturity) < 0.5
    assert qs < r + 0.5 * sigma * sigma


def test_ex10_values():
    assert ex10_threshold_q(0.05, 0.2, 0.25) == pytest.approx(
        0.062149, abs=1e-6
    )
    assert ex10_threshold_q(0.05, 0.2, 1.0) == pytest.approx(0.055644, abs=1e-6)
    assert ex10_threshold_q(0.05, 0.2, 5.0) == pytest.approx(0.042822, abs=1e-6)
    assert ex10_threshold_q_approx(0.05, 0.2, 0.25) == pytest.approx(
        ex10_threshold_q(0.05, 0.2, 0.25), abs=1e-4
    )


def test_ex11():
    put = ex11_put_bs()
    assert put == pytest.approx(3.918999, abs=1e-6)
    mid, simp = ex11_run(Rule.MIDPOINT), ex11_run(Rule.SIMPSON)
    for steps in (mid, simp):
        assert steps[-1].value == pytest.approx(put, abs=5e-7)
        for a, b in zip(steps[2:], steps[3:]):
            assert abs(b.error) < abs(a.error)
    assert (mid[-1].n, simp[-1].n) == (8192, 128)
    assert observed_order(
        ex11_integrand, 0.0, 50.0, put, 64, Rule.MIDPOINT
    ) == pytest.approx(2.0, abs=0.1)
    assert observed_order(
        ex11_integrand, 0.0, 50.0, put, 32, Rule.SIMPSON
    ) == pytest.approx(4.0, abs=0.2)


@pytest.mark.parametrize("c", CASES)
def test_ex12(c):
    spot, strike, maturity, sigma, r, q = c
    numeric = rn_price_above(lambda x: (x - strike) ** 2, strike, c)
    assert ex12_squared_call(
        spot, strike, maturity, sigma, r, q
    ) == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("c", CASES)
def test_ex13(c):
    spot, strike, maturity, sigma, r, q = c
    assert ex13_power_call(
        spot, strike, maturity, sigma, r, 1.0, q
    ) == pytest.approx(bs_call(spot, strike, maturity, sigma, r, q), abs=1e-10)
    for alpha in (0.5, 1.5, 2.0):
        strike_alpha = strike**alpha
        numeric = rn_price_above(lambda x: x**alpha - strike_alpha, strike, c)
        assert ex13_power_call(
            spot, strike_alpha, maturity, sigma, r, alpha, q
        ) == pytest.approx(numeric, rel=1e-6, abs=1e-9)


def test_ex14():
    spot0, strike, maturity, sigma, r = 50, 52, 0.75, 8.0, 0.04
    forward, s = ex14_book_forward(spot0, r, maturity), sigma * math.sqrt(
        maturity
    )
    numeric = math.exp(-r * maturity) * simpson_rule(
        lambda z: np.maximum(forward + s * z - strike, 0.0) * _phi(z),
        (strike - forward) / s,
        12.0,
        20000,
    )
    assert ex14_call(spot0, strike, maturity, sigma, r) == pytest.approx(
        numeric, abs=1e-8
    )
    assert ex14_call(spot0, strike, maturity, sigma, r) - ex14_put_parity(
        spot0, strike, maturity, sigma, r
    ) == pytest.approx(spot0 - strike * math.exp(-r * maturity))
    assert ex14_put_parity(
        spot0, strike, maturity, sigma, r
    ) - ex14_put_forward(forward, strike, maturity, sigma, r) == pytest.approx(
        math.exp(-r * maturity) * forward - spot0
    )
    forward_martingale = spot0 * math.exp(r * maturity)
    assert ex14_call_forward(
        forward_martingale, strike, maturity, sigma, r
    ) - ex14_put_forward(
        forward_martingale, strike, maturity, sigma, r
    ) == pytest.approx(
        spot0 - strike * math.exp(-r * maturity)
    )


@pytest.mark.parametrize("c", CASES)
def test_ex15(c):
    spot, strike, maturity, sigma, r, q = c
    v1 = rn_price_above(
        lambda x: 1.0 / x, 1e-12 * spot, c
    )  # whole real line, effectively
    assert ex15_inverse_claim(spot, maturity, sigma, r, q) == pytest.approx(
        v1, rel=1e-9
    )
    v2 = rn_price_above(lambda x: 1.0 / strike - 1.0 / x, strike, c)
    assert ex15_inverse_call(
        spot, strike, maturity, sigma, r, q
    ) == pytest.approx(v2, abs=1e-10)
