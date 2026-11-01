# Chapter 6 exercises (section 6.11)

Solutions to exercises 1-18 of Stefanica, *A Primer for the Mathematics of
Financial Engineering*, 2nd ed., section 6.11 (pages 200-205).

Each exercise has one C++ header in `include/p4mfe/ch06/exercises/`
(namespace `p4mfe::ch06::ex`) and a Python module with the same name in
`python/src/p4mfe/ch06_taylor_bonds_atm/exercises/`. Every header starts
with the exercise statement and the full derivation as a comment. The
proof-style exercises (1, 3, 5-11, 13(i)) are written out there in full and
checked numerically by the tests: each identity is compared with the
function it expands, each inequality is checked on a grid, and each
"same up to O(·)" claim is checked by its rate as the variable shrinks.

The code reuses the chapter 6 library (`taylor_polynomial`, `exp_series`,
`log1p_series`, `geometric_series`, `binomial_series`, `atm_price_approx`,
`atm_put_approx`, `atm_price_exact`, `DollarRisk`, `immunize`,
`value_change_approx_convexity`, `shifted_curve_price`,
`yield_change_from_shift`), chapter 2's integration (`integrate_to_tolerance`,
`improper_integral`) and bond code (`bond_cash_flows`,
`bond_price_from_zero_rate`, `bond_yield`, `price_approx_duration[_convexity]`),
chapter 3's `bs_put` and chapter 1's finite differences.

```sh
# C++: tests and a program that prints every answer
ctest --test-dir build -R Ch06Ex
./build/cpp/ch06_taylor_bonds_atm/exercises/ch06_exercises_solutions

# Python: tests and the same printout
pytest python/tests/ch06_taylor_bonds_atm/exercises
python -m p4mfe.ch06_taylor_bonds_atm.exercises
```

| # | File | Answer |
|---|------|--------|
| 1 | `ex01_sqrt_taylor` | f(0), f′(0), f″(0), f‴(0) = 1, ½, −¼, ⅜ ⇒ 1 + x/2 − x²/8 + x³/16; error −(5/128)x⁴ + O(x⁵) |
| 2 | `ex02_exp_six_digits` | Lagrange bound 3^x x^(n+1)/(n+1)! < 0.5·10⁻⁶ first at n = 5: P₅(0.25) = 1.2840250651, so e^0.25 = 1.284025 |
| 3 | `ex03_big_o` | e^(−x) − 1/(1+x) = Σ_{k≥2} (−1)^k (1/k! − 1) x^k = −x²/2 + (5/6)x³ − ..., ≤ 2x² for \|x\| ≤ ½ |
| 4 | `ex04_exp_composite_taylor` | e^g ≈ e^(g(0)) (1 + g′(0)x + (g″(0) + g′(0)²)x²/2); e^((x+1)²) ≈ e(1 + 2x + 3x²) both ways |
| 5 | `ex05_even_series` | ln(1 − x²) = −Σ_{k≥1} x^(2k)/k, 1/(1 − x²) = Σ_{k≥0} x^(2k), \|x\| < 1 |
| 6 | `ex06_e_bounds` | with t = 1/x: ln(1+t) > 2t/(2+t) and ln(1+t) < 12t/(12 + 6t − t²), both by a derivative sign (t²/(...) > 0 and t³(24 − t)/(...) > 0) |
| 7 | `ex07_log_ratio_series` | ln((1+x)/(1−x)) = 2 Σ x^(2j+1)/(2j+1) = 2 artanh x, R = 1 |
| 8 | `ex08_cosh_series` | R = ∞ (term ratio x⁴/((2k+1)(2k+2)) → 0); the series is cosh(x²) = (e^(x²) + e^(−x²))/2 |
| 9 | `ex09_log_series_convergence` | f − Pₙ = ∫₀ˣ (−1)ⁿ (x−t)ⁿ/(1+t)^(n+1) dt; \|f − Pₙ\| ≤ \|x\|ⁿ \|ln(1+x)\| → 0 on (−1, 1) |
| 10 | `ex10_log_product_integral` | ∫₀¹ ln(1−x) ln x dx = Σ 1/(n(n+1)²) = 2 − π²/6 = 0.355065933 |
| 11 | `ex11_crr_expansion` | u, d = 1 ± σ√δt + σ²δt/2 + O(δt^(3/2)); p = ½ + (e^(rδt) − A)/(2√(A² − 1)) = ½ + ½(r/σ − σ/2)√δt + O(δt^(3/2)) |
| 12 | `ex12_atm_put_error` | (i) 5.984134, (ii) P_BS = 5.968587, error 0.2605%; (iii) (6.103) 4.814848 vs 4.886987, error 1.4761% |
| 13 | `ex13_atm_parity` | (i) C − P = S(e^(−qT) − e^(−rT)) exactly; (ii) 4.861031, error 0.5311% (vs 1.4761%) |
| 14 | `ex14_atm_put_maturity` | T = 3, 5, 10, 20: 1.126%, 1.878%, 3.763%, 7.550% (≈ σ²T/24, always too high) |
| 15 | `ex15_atm_put_dividends` | T = 1, 3, 5, 10, 20: 0.420%, 2.830%, 7.261%, 28.90%, 141.4% (negative "price" at 20 years) |
| 16 | `ex16_bond_price_change` | see table below |
| 17 | `ex17_portfolio_immunization` | (i) $49,505,000. (ii) sell 20,202.02 units of bond 1 and 139,762.40 units of bond 2 |
| 18 | `ex18_curve_shift_yield` | (i) B = 106.199524, y = 4.2511%. (ii) yield changes 0.99979·dr (table below) |

Exercise 16, B = 101, D = 1.5, C = 2.5:

| dy | 10bp | 50bp | 1% | 2% |
|---|---|---|---|---|
| (2.57) B(1 − D dy) | 100.848500 | 100.242500 | 99.485000 | 97.970000 |
| (6.73) B(1 − D dy + C dy²/2) | 100.848626 | 100.245656 | 99.497625 | 98.020500 |

Exercise 18, yield change for a parallel shift dr of the zero curve:

| dr | 10bp | 20bp | 50bp | 100bp | 200bp |
|---|---|---|---|---|---|
| yield change | 9.9979bp | 19.9957bp | 49.9893bp | 99.9784bp | 199.9562bp |

Assumptions and notes on the text:

- Exercise 2: "six decimal digits" is read as an error below 0.5·10⁻⁶ (the
  rounded value is then correct). The Lagrange bound uses e^c ≤ 3^x, so it
  does not need the value being computed.
- Exercise 6: the stated hint, (1 + 1/x)^x < e < (1 + 1/x)^(x+1), is not
  needed for the proof; the proof shows the new bounds sharpen it (both
  exponents lie strictly between x and x + 1). The right inequality holds
  for every x > 0.
- Exercise 10: the book writes the sum in (iii)-(iv) with index k and
  summand in n; the sum is over n ≥ 1.
- Exercise 11: the u, d and p expansions agree up to O(δt^(3/2)), one half
  order better than the O(δt) the exercise asks for; the tests check the
  δt^(3/2) rate.
- Exercises 12-15: the relative error is |P_BS − P_approx| / P_BS, as defined
  in exercise 12 (section 6.8.3 divides by P_approx instead). ATM means
  K = S, so the strike 40 of exercises 14-15 is also the spot.
- Exercise 16: no cash flows are given, so only the two approximations are
  computed. "Duration" is read as the duration with respect to a
  continuously compounded yield, as in (2.57) and (6.73).
- Exercise 17: the "Value" column is the price of one bond, so one unit of
  bond 1 has D$ = 1100 · 2.5 and C$ = 1100 · 12. Immunizing is read as
  zeroing both the dollar duration and the dollar convexity (6.7.2). The
  short bond positions are worth $37.04mil; the chapter's `immunize` also
  returns the cash leg (−$12.96mil) that brings the total value to zero.
  Cash has no duration or convexity, so it does not change the hedge.
- Exercise 18: the four-year bond's first coupon is in six months (no
  accrued interest); yields are continuously compounded.
