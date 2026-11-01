# Chapter 10 exercises (section 10.10)

Solutions to exercises 1-17 of Stefanica, *A Primer for the Mathematics of
Financial Engineering*, 2nd ed., section 10.10 (pages 319-324).

Each header in `include/p4mfe/ch10/exercises/` (namespace `p4mfe::ch10::ex`)
has a Python module with the same name in
`python/src/p4mfe/ch10_math_appendix/exercises/`. Every header starts with the
exercise statement and the full derivation or proof as a comment. Most of this
chapter's exercises are proofs; each one is backed by a numerical check in the
tests: exact integer identities, recursions run forward against their closed
forms, bounded or vanishing ratios, partial sums between their bounds, and
iterations converging to the computed limits. The code reuses chapter 10's
`parity`, `is_even`, `is_odd`, `sum_kj_xk`, `sum_k2_xk`, `sum_k`, `sum_k2`,
`sum_k3`, `sum_of_powers`, `sums_of_powers_recursive`,
`sum_of_powers_polynomial`, `binomial`, `recursion_terms`,
`solve_second_order`, `first_order_affine`, `interpolating_polynomial`,
`polynomial_value`, `observed_power`, `observed_power_at_infinity`,
`order_ratios`, `radius_ratio_estimate` and `radius_root_estimate`, and
chapter 5's `solve_linear`.

```sh
# C++: tests and a program that prints every answer
ctest --test-dir build -R Ch10Ex
./build/cpp/ch10_math_appendix/exercises/ch10_exercises_solutions

# Python: tests and the same printout
pytest python/tests/ch10_math_appendix/exercises
python -m p4mfe.ch10_math_appendix.exercises
```

| # | File | Answer |
|---|------|--------|
| 1 | `ex01_parity` | h(−x) = (−x)ⁱ f((−1)ʲxʲ) = (−1)ⁱ⁺ʲ h(x) since f((−1)ʲy) = (−1)ʲf(y): even if i + j is even, odd if i + j is odd. So x f(x) is even, x²f(x) odd, f(x²) even, f(x³) odd |
| 2 | `ex02_lhopital_sums` | (i) N₂‴(1) = −n(n+1)(2n+1) and ((1−x)³)‴ = −6, so S(n,2) = n(n+1)(2n+1)/6. (ii) T(n,3,x) = [x + 4x² + x³ − (n+1)³xⁿ⁺¹ + (3n³+6n²−4)xⁿ⁺² − (3n³+3n²−3n+1)xⁿ⁺³ + n³xⁿ⁺⁴]/(1−x)⁴. (iii) N₃⁗(1) = 6n²(n+1)², so S(n,3) = (n(n+1)/2)² |
| 3 | `ex03_sum_k4` | S(n,4) = n(n+1)(2n+1)(3n²+3n−1)/30 = n⁵/5 + n⁴/2 + n³/3 − n/30 |
| 4 | `ex04_sum_squares_recursion` | Shift and subtract three times: x_{n+4} − 4x_{n+3} + 6x_{n+2} − 4x_{n+1} + x_n = 0, P(z) = (z−1)⁴, so x_n is a cubic in n; the cubic through (1,1), (2,5), (3,14), (4,30) is n(n+1)(2n+1)/6 |
| 5 | `ex05_07_linear_recursions` | z³ − 2z − 1 = (z+1)(z² − z − 1): x_n = c₁(−1)ⁿ + c₂φⁿ + c₃ψⁿ, φ, ψ = (1 ± √5)/2. The initial values give c = (1, 0, 0): x_n = (−1)ⁿ |
| 6 | `ex05_07_linear_recursions` | (i) x_{n+2} = 4x_{n+1} − 3x_n, x₁ = 5. (ii) roots 1, 3: x_n = 2·3ⁿ − 1 |
| 7 | `ex05_07_linear_recursions` | (i) x_{n+2} = 4x_{n+1} − 3x_n + 1, then x_{n+3} = 5x_{n+2} − 7x_{n+1} + 3x_n, x₂ = 18. (ii) (z−1)²(z−3): x_n = −5/4 − n/2 + (9/4)3ⁿ = (3ⁿ⁺² − 2n − 5)/4 |
| 8 | `ex08_double_root` | Σ aᵢ C(n+i)λⁿ⁺ⁱ = Cnλⁿ P(λ) + Cλⁿ⁺¹ P′(λ) = 0 when P(λ) = P′(λ) = 0 (double root). Checked with P = (z−2)²(z+1): n2ⁿ solves x_{n+3} = 3x_{n+2} − 4x_n |
| 9 | `ex09_10_big_o` | \|f + g\| ≤ (C_f + C_g)\|x\|ⁿ near 0, so O + O = O; (f + g)/xⁿ = f/xⁿ + g/xⁿ → 0, so o + o = o |
| 10 | `ex09_10_big_o` | Σk²/n³ = 1/3 + 1/(2n) + 1/(6n²) ≤ 1, (Σk² − n³/3)/n² = 1/2 + 1/(6n) ≤ 2/3; Σk³/n⁴ = 1/4 + 1/(2n) + 1/(4n²) ≤ 1, (Σk³ − n⁴/4)/n³ = 1/2 + 1/(4n) ≤ 3/4 |
| 11 | `ex11_binomial_identities` | a = b = 1 gives 2ⁿ; adding a = −1, b = 1 (sum 0) gives 2Σ C(n,2k) = 2ⁿ; d/da (a+1)ⁿ at a = 1 gives Σ jC(n,j) = n2ⁿ⁻¹ |
| 12 | `ex12_13_harmonic_series` | Σ1/k² < 2 − 1/n (from 1/k² < 1/(k−1) − 1/k) converges. ln n + 1/n < H_n < ln n + 1 (n ≥ 2) → ∞. ln ln n − ln ln 2 + 1/(n ln n) < Σ_{k=2}^{n} 1/(k ln k) < ln ln n − ln ln 2 + 1/(2 ln 2) → ∞ |
| 13 | `ex12_13_harmonic_series` | x_n = H_n − ln n decreases (x_{n+1} − x_n = 1/(n+1) − ∫ₙⁿ⁺¹dx/x < 0) and y_n = H_n − ln(n+1) increases, with x_n − y_n → 0: the limit γ is in [1 − ln 2, 3/2 − ln 2] = [0.307, 0.807]. Numerically x_n − γ ≈ 1/(2n) |
| 14 | `ex14_power_series_radius` | \|a_k/a_{k+1}\| = (k+1)ln(k+1)/(k ln k) → 1, so R = 1. Diverges at x = 1 (exercise 12), converges conditionally at x = −1 (alternating, 1/(k ln k) ↓ 0). Converges exactly on [−1, 1); the sum at −1 is 0.526412 |
| 15 | `ex15_17_nested_limits` | (1 + √(1 + 4a))/2 (the iterates √a, √(a + √a), … increase and stay below it) |
| 16 | `ex15_17_nested_limits` | (a + √(a² + 4))/2 (successive differences shrink by at least 1/(1 + a²)) |
| 17 | `ex15_17_nested_limits` | (i) x = √2 (x^b = b with b = 2 means x = 2^(1/2)). (ii) x = b^(1/b) is largest at b = e: the largest x is e^(1/e) = 1.444668 and the largest b is e |

Assumptions and notes on the text:

- The photos show the exercises but not the chapter text, so the equation
  numbers are read as follows. (10.31) is T(n, j+1, x) = x d/dx T(n, j, x), as
  exercise 2 (ii) itself says. (10.26) is the telescoping recursion
  (n+1)^(i+1) − 1 = Σ_{j=0}^{i} C(i+1, j) S(n, j), the one chapter 10's
  `sums_of_powers_recursive` implements, and (10.14)-(10.16) are the formulas
  for S(n,1), S(n,2) and S(n,3).
- Exercise 2: l'Hôpital's rule needs the numerator to vanish to the same
  order as (1 − x)^(j+1) at x = 1. The tests check N_j(1) = N_j′(1) = … =
  N_j^(j)(1) = 0 for every n ≤ 200, and the code evaluates the (j+1)-st
  derivatives in exact integer arithmetic, so the same recipe also gives S(n,1)
  and S(n,4). The book's closed form for N₂ has distinct exponents only for
  n ≥ 3 (for n = 1, 2 some powers coincide and the coefficients add up); the
  comparison with the book's formula uses n ≥ 3.
- Exercise 5 asks for "the general form"; both are given: the general
  solution c₁(−1)ⁿ + c₂φⁿ + c₃ψⁿ, and x_n = (−1)ⁿ for the given initial values.
- Exercise 9: n > 0 need not be an integer; for non-integer n, xⁿ is read as
  |x|ⁿ. The numerical illustration uses x > 0 and n = 2.
- Exercise 12: the book states ln(n) + 1/n < H_n < ln(n) + 1 for all n ≥ 1, but
  at n = 1 all three are equal to 1. The strict inequalities hold for n ≥ 2;
  the proof uses n − 1 ≥ 1 intervals.
- Exercise 14: "the points where |x| = R" are taken to be the real points
  x = ±1.
- Exercises 15-17: an infinite expression is defined as the limit of its
  truncations, x₁ = √a (15), x₁ = a (16), t₁ = x (17), each next level
  wrapping the previous one.
- Exercise 17 (ii): the largest b is taken over all x > 0 for which the tower
  converges. For x > e^(1/e), x^t > t for every t, so the tower increases
  without bound (the test sees it pass 10⁶ for x = e^(1/e) + 0.001). For
  x < 1, any limit b solves b = x^b < 1. At x = e^(1/e) the tower converges
  slowly, e − t_n ≈ 2e/n.
- Tests compare integer results exactly (C++ 64-bit integers or doubles below
  2⁵³, Python integers) and floating-point results with tolerances well above
  round-off. They were also run with clang at -O0 and with
  `-march=native -ffp-contract=fast` (FMA contraction), with the same results.
