"""Chapter 5 exercises (section 5.7), mirroring ``cpp/ch05_newton_bootstrapping/exercises``.

One module per exercise, with the same names as the C++ headers:

- ``ex01_bond_yield``                  1  yield by Newton's method, duration and convexity
- ``ex02_zero_curve_bond``             2  bond priced off a zero rate curve: price, yield, durations, convexity
- ``ex03_implied_vol_limits``          3  limits and monotonicity of the implied vol equation; when it has a root
- ``ex04_implied_vol_methods``         4  implied vol by bisection, secant and Newton; ATM approximation (6.104)
- ``ex05_central_difference_newton``   5  approximate Newton with central vs forward difference Jacobians
- ``ex06_atm_delta_dividend``          6  dividend yield with Delta(ATM call) = 0.5
- ``ex07_treasury_bootstrap``          7  Treasury bootstrap, continuous vs semiannual compounding
- ``ex08_bootstrap_overnight``         8  bootstrap with an overnight rate, linear on given intervals
- ``ex09_bootstrap_pricing_error``     9  bootstrap, then price the 3-year bond off the curve

Full derivations are in the C++ headers and in ``cpp/ch05_newton_bootstrapping/exercises/README.md``.
Run ``python -m p4mfe.ch05_newton_bootstrapping.exercises`` to print all answers.
"""
