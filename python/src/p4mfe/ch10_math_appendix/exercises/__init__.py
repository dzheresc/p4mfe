"""Chapter 10 exercises (section 10.10), mirroring ``cpp/ch10_math_appendix/exercises``.

One module per C++ header, with the same names:

- ``ex01_parity``                 1     parity of x^i f(x^j) for odd f
- ``ex02_lhopital_sums``          2     S(n, 2), S(n, 3) as T(n, j, 1) by l'Hopital's rule
- ``ex03_sum_k4``                 3     S(n, 4) from the telescoping recursion
- ``ex04_sum_squares_recursion``  4     S(n, 2) from a recursion with root 1 of multiplicity 4
- ``ex05_07_linear_recursions``   5-7   closed forms of three linear recursions
- ``ex08_double_root``            8     a double root lambda gives the solution C n lambda^n
- ``ex09_10_big_o``               9-10  O + O = O, o + o = o; orders of sum k^2, sum k^3
- ``ex11_binomial_identities``    11    sum C(n, j), sum C(n, 2k), sum j C(n, j)
- ``ex12_13_harmonic_series``     12-13 sum 1/k^2, sum 1/k, sum 1/(k ln k); H_n - ln n -> gamma
- ``ex14_power_series_radius``    14    sum x^k / (k ln k): R = 1, converges on [-1, 1)
- ``ex15_17_nested_limits``       15-17 nested radical, continued fraction, power tower

Full derivations are in the C++ headers and in ``cpp/ch10_math_appendix/exercises/README.md``.
Run ``python -m p4mfe.ch10_math_appendix.exercises`` to print all answers.
"""
