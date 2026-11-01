import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.calculus import derivative
from p4mfe.ch02_integration_bonds.integration import improper_integral
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_d1_d2, bs_greeks, bs_price, bs_put
from p4mfe.ch03_probability_black_scholes.normal_rv import lognormal_mean, lognormal_variance
from p4mfe.ch03_probability_black_scholes.probability import ContinuousDistribution
from p4mfe.ch04_lognormal_risk_neutral.asset_model import (
    log_price_distribution,
    log_return_distribution,
    price_cdf,
    price_confidence_interval,
    price_from_normal,
    price_mean,
    price_median,
    price_pdf,
    price_quantile,
    price_variance,
)
from p4mfe.ch04_lognormal_risk_neutral.in_the_money import (
    OptionType,
    asset_or_nothing,
    cash_or_nothing,
    exercise_probabilities,
    prob_between,
    prob_itm,
)
from p4mfe.ch04_lognormal_risk_neutral.risk_neutral import (
    monte_carlo_price,
    risk_neutral_expectation,
    risk_neutral_price,
    rn_expected_payoff,
)
from p4mfe.normal import norm_cdf

S0, K, T, sigma, r, q, mu = 40.0, 42.0, 0.75, 0.3, 0.04, 0.01, 0.10
CALL, PUT = OptionType.CALL, OptionType.PUT


def call_payoff(s):
    return np.maximum(s - K, 0.0)


# 4.4 -----------------------------------------------------------------------
def test_asset_moments_and_median():
    L = log_price_distribution(S0, T, sigma, mu, q)
    assert price_mean(S0, T, mu, q) == pytest.approx(lognormal_mean(L.mean, L.stddev), rel=1e-14)
    assert price_variance(S0, T, sigma, mu, q) == pytest.approx(lognormal_variance(L.mean, L.stddev), rel=1e-13)
    assert price_cdf(price_median(S0, T, sigma, mu, q), S0, T, sigma, mu, q) == pytest.approx(0.5, abs=1e-15)
    assert price_median(S0, T, sigma, mu, q) < price_mean(S0, T, mu, q)
    R = log_return_distribution(T, sigma, mu, q)
    assert R.mean == pytest.approx(L.mean - math.log(S0), abs=1e-15)


def test_asset_density_integrates_to_moments():
    S = ContinuousDistribution(lambda x: price_pdf(x, S0, T, sigma, mu, q), 0.0, math.inf)
    assert S.mean() == pytest.approx(price_mean(S0, T, mu, q), abs=1e-7)
    assert S.cdf(45.0) == pytest.approx(price_cdf(45.0, S0, T, sigma, mu, q), abs=1e-8)


def test_confidence_interval():
    rg = price_confidence_interval(0.95, S0, T, sigma, mu, q)
    assert price_cdf(rg.hi, S0, T, sigma, mu, q) - price_cdf(rg.lo, S0, T, sigma, mu, q) == pytest.approx(0.95, abs=1e-13)
    m = price_median(S0, T, sigma, mu, q)
    assert rg.lo * rg.hi == pytest.approx(m * m, rel=1e-13)
    assert rg.hi - m > m - rg.lo
    assert rg.lo == pytest.approx(m * math.exp(-1.959963984540054 * sigma * math.sqrt(T)), rel=1e-13)
    assert price_quantile(0.975, S0, T, sigma, mu, q) == pytest.approx(rg.hi, rel=1e-13)


# 4.5 -----------------------------------------------------------------------
@pytest.mark.parametrize("option_type", [CALL, PUT])
def test_closed_form_is_black_scholes(option_type):
    assert math.exp(-r * T) * rn_expected_payoff(option_type, S0, K, T, sigma, r, q) == pytest.approx(
        bs_price(option_type, S0, K, T, sigma, r, q), abs=1e-13)


def test_integral_of_payoff_is_black_scholes():
    assert risk_neutral_price(call_payoff, S0, T, sigma, r, q) == pytest.approx(bs_call(S0, K, T, sigma, r, q), abs=1e-7)
    put = lambda s: np.maximum(K - s, 0.0)  # noqa: E731
    assert risk_neutral_price(put, S0, T, sigma, r, q) == pytest.approx(bs_put(S0, K, T, sigma, r, q), abs=1e-7)
    assert risk_neutral_expectation(lambda s: s, S0, T, sigma, r, q) == pytest.approx(S0 * math.exp((r - q) * T), abs=1e-7)


def test_derivation_restricts_to_exercise_region():
    d1, d2 = bs_d1_d2(S0, K, T, sigma, r, q)
    assert price_from_normal(-d2, S0, T, sigma, r, q) == pytest.approx(K, rel=1e-14)
    phi = lambda z: np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)  # noqa: E731

    def weighted(z):
        p = phi(z)
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(p == 0.0, 0.0, price_from_normal(z, S0, T, sigma, r, q) * p)

    asset_part = improper_integral(weighted, -d2, math.inf, 1e-10).value
    cash_part = improper_integral(phi, -d2, math.inf, 1e-10).value
    assert asset_part == pytest.approx(S0 * math.exp((r - q) * T) * norm_cdf(d1), abs=1e-8)
    assert cash_part == pytest.approx(norm_cdf(d2), abs=1e-9)


@pytest.mark.parametrize("antithetic", [False, True])
def test_monte_carlo_within_standard_errors(antithetic):
    bs = bs_call(S0, K, T, sigma, r, q)
    mc = monte_carlo_price(call_payoff, S0, T, sigma, r, q, 200_000, seed=7, antithetic=antithetic)
    assert 0 < mc.std_error < 0.03
    assert abs(mc.price - bs) < 4 * mc.std_error


def test_real_world_drift_does_not_price():
    wrong = math.exp(-r * T) * risk_neutral_expectation(call_payoff, S0, T, sigma, mu, q)
    assert wrong > bs_call(S0, K, T, sigma, r, q) + 0.1


# 4.6, 4.7 ------------------------------------------------------------------
def test_itm_probabilities():
    d1, d2 = bs_d1_d2(S0, K, T, sigma, r, q)
    assert prob_itm(CALL, S0, K, T, sigma, r, q) == pytest.approx(norm_cdf(d2), abs=1e-15)
    assert prob_itm(PUT, S0, K, T, sigma, r, q) == pytest.approx(norm_cdf(-d2), abs=1e-15)
    d2_mu = d2 + (mu - r) * math.sqrt(T) / sigma
    assert prob_itm(CALL, S0, K, T, sigma, mu, q) == pytest.approx(norm_cdf(d2_mu), abs=1e-14)
    assert prob_between(35, 45, S0, T, sigma, mu, q) == pytest.approx(
        price_cdf(45, S0, T, sigma, mu, q) - price_cdf(35, S0, T, sigma, mu, q), abs=1e-15)


@pytest.mark.parametrize("option_type,sign", [(CALL, 1.0), (PUT, -1.0)])
def test_option_is_asset_minus_cash_digital(option_type, sign):
    aon = asset_or_nothing(option_type, S0, K, T, sigma, r, q)
    con = cash_or_nothing(option_type, S0, K, T, sigma, r, q)
    assert sign * (aon - K * con) == pytest.approx(bs_price(option_type, S0, K, T, sigma, r, q), abs=1e-13)


def test_digital_parities():
    assert cash_or_nothing(CALL, S0, K, T, sigma, r, q) + cash_or_nothing(PUT, S0, K, T, sigma, r, q) == pytest.approx(
        math.exp(-r * T), abs=1e-15)
    assert asset_or_nothing(CALL, S0, K, T, sigma, r, q) + asset_or_nothing(PUT, S0, K, T, sigma, r, q) == pytest.approx(
        S0 * math.exp(-q * T), abs=1e-13)


def test_financial_meaning_of_nd1_nd2():
    p = exercise_probabilities(S0, K, T, sigma, r, q)
    assert p.n_d1 > p.n_d2
    assert p.n_d2 == pytest.approx(math.exp(r * T) * cash_or_nothing(CALL, S0, K, T, sigma, r, q), abs=1e-15)
    fwd = S0 * math.exp((r - q) * T)
    assert p.n_d1 == pytest.approx(math.exp(r * T) * asset_or_nothing(CALL, S0, K, T, sigma, r, q) / fwd, abs=1e-14)
    assert bs_greeks(CALL, S0, K, T, sigma, r, q).delta == pytest.approx(math.exp(-q * T) * p.n_d1, abs=1e-15)
    dC_dK = derivative(lambda k: bs_call(S0, k, T, sigma, r, q), K)
    assert dC_dK == pytest.approx(-math.exp(-r * T) * p.n_d2, abs=1e-8)
