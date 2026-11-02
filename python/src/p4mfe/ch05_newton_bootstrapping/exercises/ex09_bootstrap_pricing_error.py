"""Exercise 9: bootstrap from the bills and the 2, 5, 10-year bonds; price the 3-year bond.

(i) r(0, t) at 0.5, 1, 2, 5, 10: 1.0900%, 1.3900%, 1.2099%, 2.6824%, 3.7372%.
(ii) 3-year 4.5% bond: model 108.193042 vs market 107 18/32 = 107.5625, relative error +0.586%.

Full derivation: cpp/ch05_newton_bootstrapping/exercises/include/p4mfe/ch05/exercises/ex09_bootstrap_pricing_error.hpp
"""

from __future__ import annotations

from p4mfe.ch02_integration_bonds.bonds import BondQuote, ZeroCurve, bond_cash_flows, bond_price_from_zero_rate

from ..bootstrapping import bootstrap_zero_curve, zero_rate_linear

EX09_QUOTES = (
    BondQuote(0.5, 0.0, 99.4565),
    BondQuote(1.0, 0.0, 98.6196),
    BondQuote(2.0, 0.02, 101.0 + 17.5 / 32.0),
    BondQuote(5.0, 0.03125, 102.0 + 8.0 / 32.0),
    BondQuote(10.0, 0.04, 103.0 + 8.5 / 32.0),
)
EX09_THREE_YEAR = BondQuote(3.0, 0.045, 107.0 + 18.0 / 32.0)


def ex09_curve() -> ZeroCurve:
    return bootstrap_zero_curve(EX09_QUOTES, 2)


def ex09_model_price(c: ZeroCurve) -> float:
    cf = bond_cash_flows(
        EX09_THREE_YEAR.maturity, EX09_THREE_YEAR.coupon_rate, 2
    )
    return bond_price_from_zero_rate(cf, lambda t: zero_rate_linear(c, t))


def ex09_relative_error(c: ZeroCurve) -> float:
    return (ex09_model_price(c) - EX09_THREE_YEAR.price) / EX09_THREE_YEAR.price
