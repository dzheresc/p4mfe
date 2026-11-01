"""Chapter 7 exercises (section 7.6), mirroring ``cpp/ch07_finite_differences_bs_pde/exercises``.

One module per C++ header, with the same names:

- ``ex01_02_difference_orders``     1-2  central difference of order 4, forward difference of order 2
- ``ex03_04_stencils``              3-4  one-sided second order f'(a); central f^(4)(a)
- ``ex05_nonsymmetric_stencils``    5    f'(a), f''(a) from non-symmetric points
- ``ex06_butterfly_spread``         6    butterflies, C_KK, Arrow-Debreu payoff
- ``ex07_bull_spread``              7    bull spreads, -C_K, cash-or-nothing payoff
- ``ex08_butterfly_delta``          8    the scaled butterfly payoff as a nascent delta
- ``ex09_euler_convergence``        9    forward Euler for y' = y is first order convergent
- ``ex10_bvp_linear_system``        10   central difference BVP, the 5 x 5 system
- ``ex11_first_order_system``       11   y'' - 2y' + x^2 y = 0 as Y' = f(x, Y)
- ``ex12_gamma_theta``              12   1 + sigma^2 S^2/2 Gamma/Theta
- ``ex13_14_bs_pde_solutions``      13-14 the call and the forward solve the Black-Scholes PDE

Full derivations are in the C++ headers and in ``cpp/ch07_finite_differences_bs_pde/exercises/README.md``.
Run ``python -m p4mfe.ch07_finite_differences_bs_pde.exercises`` to print all answers.
"""
