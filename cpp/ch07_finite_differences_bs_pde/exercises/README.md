# Chapter 7 exercises (section 7.6)

Solutions to exercises 1-14 of Stefanica, *A Primer for the Mathematics of
Financial Engineering*, 2nd ed., section 7.6 (pages 232-236).

Each header in `include/p4mfe/ch07/exercises/` (namespace `p4mfe::ch07::ex`)
has a Python module with the same name in
`python/src/p4mfe/ch07_finite_differences_bs_pde/exercises/`. Every header
starts with the exercise statement and the full derivation or proof as a
comment. Most of this chapter's exercises are proofs; each one is backed by a
numerical check in the tests: observed orders of convergence, leading error
constants, exactness on polynomials, limits of option spreads, and PDE
residuals. The code reuses chapter 7's `central_difference`,
`forward_difference`, `second_central_difference`, `euler_forward`,
`max_error`, `solve_tridiagonal`, `solve_bvp`, `bs_pde_residual_fd` and
`forward_contract_value`, chapter 3's `bs_call`, `bs_d1_d2`, `bs_greeks` and
`bs_pde_residual`, and chapter 2's `simpson_rule`.

```sh
# C++: tests and a program that prints every answer
ctest --test-dir build -R Ch07Ex
./build/cpp/ch07_finite_differences_bs_pde/exercises/ch07_exercises_solutions

# Python: tests and the same printout
pytest python/tests/ch07_finite_differences_bs_pde/exercises
python -m p4mfe.ch07_finite_differences_bs_pde.exercises
```

| # | File | Answer |
|---|------|--------|
| 1 | `ex01_02_difference_orders` | f = (x³ − 6)eˣ has f‴(0) = 0 and f⁽⁵⁾(0) = 54, so (f(h) − f(−h))/(2h) − f′(0) = 0.45 h⁴ + O(h⁶): fourth order (observed 4.000) |
| 2 | `ex01_02_difference_orders` | N″(0) = 0, so (N(h) − N(0))/h − N′(0) = −φ(0)h²/6 + O(h⁴): second order (observed 2.000) |
| 3 | `ex03_04_stencils` | f′(a) ≈ (−3f(a) + 4f(a+h) − f(a+2h))/(2h), error −h²f‴(a)/3 |
| 4 | `ex03_04_stencils` | f⁽⁴⁾(a) ≈ (f(a−2h) − 4f(a−h) + 6f(a) − 4f(a+h) + f(a+2h))/h⁴, error h²f⁽⁶⁾(a)/6: second order |
| 5 | `ex05_nonsymmetric_stencils` | (i) f′(a) ≈ (3f(a) − 4f(a−h) + f(a+2h))/(6h), error h²f‴/3. (ii) f″(a) ≈ (2f(a−h) − 3f(a) + f(a+2h))/(3h²), error hf‴/3. (iii)-(iv) general β, γ below |
| 6 | `ex06_butterfly_spread` | (i) same payoff ⇒ same value. (ii) [C(K+x) − 2C(K) + C(K−x)]/x² = C_KK + x²C_KKKK/12 + … → C_KK = e^(−rT)φ(d2)/(Kσ√T). (iii) payoff/x = (1 − \|S−K\|/x)⁺ → 1_{S=K} |
| 7 | `ex07_bull_spread` | (i) (C(K) − C(K+x))/x = −C_K − xC_KK/2 + … → −C_K = e^(−rT)N(d2), the cash-or-nothing call. (ii) payoff/x → 1_{S>K} |
| 8 | `ex08_butterfly_delta` | f(S) = (x − \|S−K\|)⁺/x² ≥ 0 has ∫f = 1 and support [K−x, K+x], so \|∫fg − g(K)\| ≤ max_{\|S−K\|≤x} \|g(S) − g(K)\| → 0; for smooth g, ∫fg = g(K) + g″(K)x²/12 + O(x⁴) |
| 9 | `ex09_euler_convergence` | yᵢ = (1+h)ⁱ; max\|eᵢ\| = e − (1+1/n)ⁿ = e/(2n) − 11e/(24n²) + …: n·max\|eᵢ\| = 1.2454, 1.3468, 1.3579 for n = 10, 100, 1000 (→ e/2 = 1.3591) |
| 10 | `ex10_bvp_linear_system` | (3i² + i/2)y_{i−1} + (1 − 6i²)yᵢ + (3i² − i/2)y_{i+1} = 0; A and b below; Y = (1.385311, 1.370622, 1.228661, 1.021815, 0.774770) |
| 11 | `ex11_first_order_system` | Y = (y, y′), Y′ = (y′, y″) = (y′, 2y′ − x²y) = [[0, 1], [−x², 2]]Y; checked with RK4 and forward Euler on the system |
| 12 | `ex12_gamma_theta` | (i) r = q = 0: ½σ²S²Γ = −Θ (calls and puts). (ii) q = 0: with A = Ke^(−rτ)σφ(d2)/(2√τ), B = rKe^(−rτ)N(d2), ½σ²S²Γ = A and Θ = −(A + B), so 1 + ½σ²S²Γ/Θ = B/(A+B) = 1/(1 + σN′(d2)/(2r√τ N(d2))) |
| 13 | `ex13_14_bs_pde_solutions` | Θ + ½σ²S²Γ + (r−q)SΔ − rC: the φ(d1) terms cancel, the Se^(−qτ)N(d1) terms sum to (q + r − q − r) = 0, the Ke^(−rτ)N(d2) terms to (−r + r) = 0 |
| 14 | `ex13_14_bs_pde_solutions` | f_t = qSe^(−qτ) − rKe^(−rτ), f_S = e^(−qτ), f_SS = 0; the PDE sums to 0 for every σ. f = C − P, so the put solves the PDE too |

Exercise 5, general points β < a < γ, u = a − β = C(γ − a), v = γ − a:

| | Formula | Error (approximation − exact) |
|---|---|---|
| f′(a) | [u²f(γ) − v²f(β) − (u² − v²)f(a)] / (uv(γ − β)) | uv f‴(a)/6 = O(\|γ − a\|²) |
| f″(a) | 2[u f(γ) + v f(β) − (γ − β) f(a)] / (uv(γ − β)) | (v − u) f‴(a)/3 = (β + γ − 2a) f‴(a)/3 |

The f″ weights are the only ones that match f, f′ and f″, and their f‴
coefficient vanishes only for u = v, i.e. a = (β + γ)/2. For f = (x − a)³ the
error is exactly 2(β + γ − 2a), proportional to |γ − a|, so no second order
formula exists in the non-symmetric case.

Exercise 10, n = 6:

```
      | -5    2.5                      |        | -3.5   |
      | 13   -23    11                 |        |  0     |
  A = |      28.5  -53    25.5         |,  b =  |  0     |
      |             50   -95    46     |        |  0     |
      |                  77.5  -149    |        | -36.25 |
```

Assumptions and notes on the text:

- Exercises 6-7 have no numbers. The numerical checks use S = 52, K = 50,
  T = 0.5, σ = 25%, r = 4%, q = 1% (Black-Scholes call prices). There
  C_KK = 0.0431994 and e^(−rT)N(d2) = 0.5748023.
- Exercise 8: "the value of a call with strike K and maturity T on an asset
  with spot price S" is read as the value at maturity, i.e. the payoff
  (S − K)⁺ with S = S(T). Only then is f a function of S that concentrates at
  K. With the Black-Scholes value before maturity, f(S) tends to C_KK(K),
  whose integral against g is not g(K).
- Exercise 9: the hint's remainders O(h³) hold for fixed i. Uniformly in
  i ≤ n they are O(h²), still o(h), which is what the bound e/(2n) + o(1/n)
  needs. The bound is sharp: the largest error is |eₙ| = e − (1 + 1/n)ⁿ.
- Exercise 10: the boundary value problem has no solution. The ODE is an
  Euler equation; y = xᵐ solves it iff (3m − 1)(m − 1) = 0, so every
  solution is c₁x + c₂x^(1/3), which is 0 at x = 0 and never 1. The
  discrete system for each n is still uniquely solvable (the exercise asks
  only for A and b), but its solutions do not converge as n grows: y at
  x = ½ is 1.229, 1.479, 1.797, 2.198, 2.704 for n = 6, 12, 24, 48, 96.
- Exercise 12: τ = T − t, Θ = ∂V/∂t. For r < 0 the formula in (ii) still holds
  (checked for r = −1%); the value is in (0, 1) only for r > 0.
- Tests use step sizes large enough (h ≥ 0.05 for the order checks) that
  truncation error dominates round-off on any IEEE double machine, and
  tolerances with margin. They were also run with clang at -O0 and with
  `-march=native -ffp-contract=fast` (FMA contraction), with the same results.
