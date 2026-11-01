# Chapter 5 exercises (section 5.7)

Solutions to exercises 1-9 of Stefanica, *A Primer for the Mathematics of
Financial Engineering*, 2nd ed., section 5.7 (pages 157-160).

Each exercise has one C++ header in `include/p4mfe/ch05/exercises/`
(namespace `p4mfe::ch05::ex`) and a Python module with the same name in
`python/src/p4mfe/ch05_newton_bootstrapping/exercises/`. Every header starts
with the exercise statement and the full derivation as a comment; the
proof-style parts (exercise 3, 6, 7(ii)) are written out there and checked
numerically by the tests. The code reuses the chapter 5 solvers
(`bisection`, `newton`, `secant`, `approximate_newton_nd`,
`bond_yield_newton`, `bootstrap_zero_curve`), the chapter 2 bond code and
the chapter 3 Black-Scholes formulas. The tests check every answer
independently: yields reprice their bonds, durations and convexities match
finite differences of B(y), every bootstrapped curve reprices all of its
bonds, and the implied volatilities reprice their calls.

Conventions: face value 100; yields and zero rates are continuously
compounded unless stated; k months is k/12 years.

```sh
# C++: tests and a program that prints every answer
ctest --test-dir build -R Ch05Ex
./build/cpp/ch05_newton_bootstrapping/exercises/ch05_exercises_solutions

# Python: tests and the same printout
pytest python/tests/ch05_newton_bootstrapping/exercises
python -m p4mfe.ch05_newton_bootstrapping.exercises
```

| # | File | Answer |
|---|------|--------|
| 1 | `ex01_bond_yield` | y = 3.3401% (Newton from 0.1, 5 steps), D = 4.642735, C = 22.573118 |
| 2 | `ex02_zero_curve_bond` | B = 104.173911, y = 5.6792%, D = 1.890108 (the modified duration for a continuous yield), C = 3.689501. With a semiannual yield 5.7606% the modified duration is 1.837191 |
| 3 | `ex03_implied_vol_limits` | (i) d1 → ∞, d2 → −∞, f → Se^(−qT) − C. (ii) d1, d2 → −∞ / 0 / ∞ as F < / = / > K; f → −C (F ≤ K) or Se^(−qT) − Ke^(−rT) − C (F > K). (iii) f′ = vega > 0. (iv) a root exists iff max(Se^(−qT) − Ke^(−rT), 0) < C < Se^(−qT): the no-arbitrage range (3.99) without its end points |
| 4 | `ex04_implied_vol_methods` | (i) σ_imp = 0.397049: bisection 30 steps, secant (0.49, 0.5) 3 steps, Newton (0.5) 3 steps. (ii) σ_approx = 0.396672, relative error 9.49·10⁻⁴ |
| 5 | `ex05_central_difference_newton` | from (1, 2, 3): (−1.690551, 1.983107, −0.884558) in 9 steps; from (2, 2, 2): a long chaotic excursion, usually ending at the exact root (−1, 3, 1), but the root reached depends on rounding (details below) |
| 6 | `ex06_atm_delta_dividend` | Δ(q) = e^(−qT)N(d1) strictly decreasing, Δ(0) > ½ > Δ(r + σ²/2) ⇒ unique q* = 6.6906% (Newton from 0, 4 steps) < r + σ²/2 = 8.5% |
| 7 | `ex07_treasury_bootstrap` | (i) continuous curve below. (ii) same discount factors, r₂ = 2(e^(r/2) − 1) > r: the semiannual curve is higher, by about r²/4 (5.4-6.9 bp) |
| 8 | `ex08_bootstrap_overnight` | r(0, t) at 0, ½, 1, 5/3, 10/3, 5: 5%, 5.0636%, 4.9370%, 5.2983%, 4.5326%, 3.2119% |
| 9 | `ex09_bootstrap_pricing_error` | (i) r(0, t) at ½, 1, 2, 5, 10: 1.0900%, 1.3900%, 1.2099%, 2.6824%, 3.7372%. (ii) 3-year bond: 108.193042 vs 107.5625, relative error +0.586% |

Exercise 7, bootstrapped zero rates (linear between maturities):

| t | 0.25 | 0.5 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| continuous | 5.2341% | 5.0636% | 4.7289% | 4.7582% | 4.6303% | 4.6772% |
| semiannual | 5.3032% | 5.1282% | 4.7853% | 4.8152% | 4.6844% | 4.7323% |
| difference (bp) | 6.91 | 6.46 | 5.63 | 5.71 | 5.40 | 5.51 |

Exercise 5, iterations with h = 10⁻⁶, tol_consec = 10⁻⁶, tol_approx = 10⁻⁹:

| x₀ | root | central | forward | exact Jacobian |
|---|---|---|---|---|
| (1, 2, 3) | (−1.690551, 1.983107, −0.884558) | 9 | 9 | 9 |
| (2, 2, 2) | (−1, 3, 1) here; see below | 41 (C++) / 127 (NumPy) | 166 / 81 | 40 / 40 |

From (2, 2, 2) the first Newton step lands at (−0.45, −5.35, 16.47) and the
iterates wander as far as |x| ≈ 400 before converging. In that chaotic phase
the O(h) vs O(h²) Jacobian error, and even the rounding differences between
the C++ and NumPy builds, change the path, so neither the iteration counts
nor the root found are reproducible: on the GitHub CI runner one of the
difference schemes ends at (−1.690551, 1.983107, −0.884558) instead. The
tests accept either root from (2, 2, 2). Near a root both schemes converge
quadratically in the same number of steps, so the central scheme, at twice
the cost per Jacobian, buys nothing here.

Assumptions and notes on the text:

- Yields in exercises 1-2 are continuously compounded (Table 5.6). The
  "modified duration" of exercise 2 for a continuous yield is the duration
  −(1/B) dB/dy; the semiannual-yield modified duration is also given.
- Exercise 4: "six decimal digits" is read as the book's suggested
  tolerances, tol_consec = 10⁻⁶ and tol_approx = 10⁻⁹. The secant method
  needs two starting points; we use x₋₁ = 0.49, x₀ = 0.5, following the
  book's example after Table 5.3 (x₋₁ = x₀ − 0.01).
- Exercises 7 and 9: no rate before the first bill is given; the curve is
  taken flat before it (no cash flow falls there). In exercise 7(ii) the
  semiannual curve is bootstrapped with linear interpolation of the
  semiannual rate, so away from the bills it differs from the converted
  continuous curve by about 10⁻⁴ bp.
- Exercise 8: the hint prints r(0, 7/6) = (3 r(0, 1) + 4 r(0, 5/3))/4. The
  linear interpolation it states gives (3 r(0, 1) + r(0, 5/3))/4, which is
  what the code uses; the tests check that (5.43) then holds.
