# Chapter 8 exercises (section 8.12)

Solutions to exercises 1-21 of Stefanica, *A Primer for the Mathematics of
Financial Engineering*, 2nd ed., section 8.12 (pages 266-270).

Each header in `include/p4mfe/ch08/exercises/` (namespace `p4mfe::ch08::ex`)
has a Python module with the same name in
`python/src/p4mfe/ch08_multivariable_barriers/exercises/`. Every header
starts with the exercise statement and the full derivation or proof as a
comment. The proofs are backed by numerical checks in the tests: monotonicity
on grids, double integrals by iterated Simpson, PDE residuals by finite
differences, a heat-equation finite difference solution with the derived
boundary condition, binomial American prices, and Greeks against finite
differences. The code reuses chapter 8's `double_integral_region`,
`polar_integral`, `change_of_variables_integral`, `jacobian_determinant`,
`heat_transform`, `solve_heat_explicit`, `interpolate`, `down_and_out_call`
(formula 8.46), `down_and_in_call`, `binomial_price` and
`early_exercise_premium`, chapter 3's `bs_call`, `bs_put`, `bs_d1_d2`,
`bs_greeks` and `implied_vol_newton`, chapter 2's `simpson_rule`, and
chapter 5's `bisection`.

```sh
# C++: tests and a program that prints every answer
ctest --test-dir build -R Ch08Ex
./build/cpp/ch08_multivariable_barriers/exercises/ch08_exercises_solutions

# Python: tests and the same printout
pytest python/tests/ch08_multivariable_barriers/exercises
python -m p4mfe.ch08_multivariable_barriers.exercises
```

| # | File | Answer |
|---|------|--------|
| 1 | `ex01_gamma_maximum` | d/dS ln Γ = −(d1 + σ√T)/(Sσ√T): Γ increases up to S* = K e^(−(r + 3σ²/2)T) (where d1 = −σ√T) and decreases after. ln Γ = −ln S − d1²/2 + c is a downward quadratic in ln S, so Γ → 0 at both ends. K = 50, T = ½, σ = 30%, r = 5%: S* = 45.582461, Γ(S*) = 0.040340 |
| 2 | `ex02_03_double_integrals` | s = x + y, t = x − y, dx dy = ½ ds dt, D ↔ 0 ≤ s ≤ 1, −s ≤ t ≤ s: ∫₀¹ ∫₋ₛˢ t/(2s) dt ds = **0** |
| 3 | `ex02_03_double_integrals` | Curves meet at x = 1, 2; inner integral 7x³ − 21x² + 14x; answer **−7/4** |
| 4 | `ex04_05_polar_areas` | det J = r: ∫₀^2π ∫₀^R r dr dθ = πR² |
| 5 | `ex04_05_polar_areas` | det J = abr: ∫₀^2π ∫₀¹ abr dr dθ = **πab** |
| 6 | `ex06_07_exp_inequalities` | ln(x)/x has its maximum at x = e, so ln(π)/π < 1/e and **e^π > π^e** (23.140693 > 22.459158) |
| 7 | `ex06_07_exp_inequalities` | (eˣ + eʸ)/2 − e^((x+y)/2) = (e^(x/2) − e^(y/2))²/2 ≥ 0, equality iff x = y |
| 8 | `ex08_gronwall` | W = M + ∫uv, V = ∫v: (W e^(−V))′ = v(u − W)e^(−V) ≤ 0, so u ≤ W ≤ M e^V. Sharp for u = M e^V |
| 9 | `ex09_10_heat_change_of_variables` | τ = 0 ↔ t = T and S = Keˣ: u(x, 0) = e^(ax) V(Keˣ, T) = K e^(ax) max(eˣ − 1, 0) (call), K e^(ax) max(1 − eˣ, 0) (put) |
| 10 | `ex09_10_heat_change_of_variables` | a = (r−q)/σ² − ½, then 1 − 2(r−q)/σ² = −2a and b = a² + 2r/σ² = ((r−q)/σ² + ½)² + 2q/σ² (see note) |
| 11 | `ex11_barrier_rebate` | u(ln(B/K), τ) = R (B/K)^a e^(bτ) for 0 ≤ τ ≤ σ²T/2: a growing Dirichlet condition at the left end of u_τ = u_xx on x > ln(B/K) |
| 12 | `ex12_13_asian_pdes` | Chain rule with y_t = −ln S/T, y_I = 1/T, y_S = (T−t)/(TS): the ln S terms cancel and the F-PDE follows |
| 13 | `ex12_13_asian_pdes` | V_I = H_R, V_S = H − R H_R, V_SS = R² H_RR/S: (8.53) = S × (8.55) |
| 14 | `ex14_itm_puts` | Table below; intrinsic > Black-Scholes for **K = 48, 3 months; K = 51, 6 and 3 months** |
| 15 | `ex15_time_value` | f = C − (S − K)⁺ has f′ = e^(−qT)N(d1) > 0 for S < K and e^(−qT)N(d1) − 1 < 0 for S > K: maximum at S = K |
| 16 | `ex16_17_down_and_out_call` | p = 2r/σ² − 1 = 1/9: V = C(40) − (35/40)^p C(30.625) = 3.853951 − 0.455067 = **3.398883** |
| 17 | `ex16_17_down_and_out_call` | r = q = 0: V = C − (B N(d1) − (SK/B) N(d2)); l'Hôpital gives lim N(d2)/B = (2/(σ√T)) lim φ(d2)/B = 0 since −d2²/2 ~ −2(ln B)²/(σ²T) beats −ln B |
| 18 | `ex18_doc_greeks` | With g = (B/S)^p, X = B²/S: Δ = Δ_C(S) + (g/S)[pC(X) + XΔ_C(X)], Γ = Γ_C(S) − (g/S²)[p(p+1)C(X) + 2(p+1)XΔ_C(X) + X²Γ_C(X)]. Ex 16 data: Δ = 0.703253, Γ = 0.018758 |
| 19 | `ex19_20_early_exercise` | (i) C = P + S − Ke^(−rT) > S − K and C > 0. (ii) C_A ≥ C > (S − K)⁺, so exercising early gives up value: never optimal |
| 20 | `ex19_20_early_exercise` | C < S e^(−qT), so C − (S − K) < K − S(1 − e^(−qT)) ≤ 0 for S ≥ K/(1 − e^(−qT)) |
| 21 | `ex21_implied_vol_skew` | σ′_imp(K) = (C′(K) + e^(−rT) N(d2)) / (S e^(−qT) √T φ(d1)), d1, d2 at σ_imp(K) |

Exercise 14, S = 40, σ = 30%, r = 5%, q = 0:

| Strike | Maturity | Black-Scholes | Intrinsic | Intrinsic > BS? |
|---|---|---|---|---|
| 45 | 6 months | 5.819577 | 5 | no |
| 45 | 3 months | 5.340299 | 5 | no |
| 48 | 6 months | 8.032528 | 8 | no |
| 48 | 3 months | 7.823400 | 8 | yes |
| 51 | 6 months | 10.486171 | 11 | yes |
| 51 | 3 months | 10.547645 | 11 | yes |

Deep in the money P → Ke^(−rT) − S < K − S: the interest on the strike
outweighs the optionality, more so for short maturities.

Exercise 20, K = 40, T = 1, σ = 30%, r = 5%: the exact crossover S* where
C(S*) = S* − K, and the sufficient bound from the proof.

| q | S* | K/(1 − e^(−qT)) |
|---|---|---|
| 5% | 56.7112 | 820.17 |
| 1% | 196.0594 | 4020.03 |
| 0.1% | 1951.7986 | 40020.00 |

For small q, S* ≈ K(1 − e^(−rT))/(1 − e^(−qT)) ≈ Kr/q.

Assumptions and notes on the text:

- Exercise 10: the printed answer b = ((r − q)/σ² + ½)² solves the system only
  for q = 0. For q > 0 it leaves −2q/σ² in the second equation. The solution is
  b = ((r − q)/σ² + ½)² + 2q/σ², which is the b of exercises 9 and 11 (the
  tests check both).
- Exercise 9 notation: V = e^(−ax−bτ)u has no factor K, so u = K·u_lib where
  the chapter library writes V = K e^(a′x+b′τ) u_lib with a′ = −a, b′ = −b.
- Exercise 11: V(B, t) = R is read as a rebate paid when the barrier is hit.
  The check solves u_τ = u_xx by explicit finite differences with the derived
  boundary condition and compares with the down-and-out call plus the
  Reiner-Rubinstein value of the rebate,
  R[(B/S)^(a+ℓ)N(z) + (B/S)^(a−ℓ)N(z − 2ℓσ√T)], ℓ = √b,
  z = ln(B/S)/(σ√T) + ℓσ√T (S = K = 40, B = 35, R = 2: 4.439140, FD
  within 2e-5).
- Exercises 12-13 have no numbers. Exercise 12 is checked on the closed-form
  geometric Asian call (K = 50, T = 1, σ = 30%, r = 5%), which at t = 0 is a
  Black-Scholes call with volatility σ/√3 and yield (r + σ²/6)/2. Exercise 13
  is checked on the exact solution H = (1 − e^(−r(T−t)))/r + e^(−r(T−t))R
  (the value of the payoff ∫₀ᵀ S dt) and, for both exercises, on the operator
  identity for an arbitrary smooth F or H.
- Exercise 15: holds for any r and q ≥ 0. Exercise 19 needs r ≥ 0 (for r < 0
  deep in-the-money calls are worth less than S − K).
- Exercise 17: the hint's e^(−rT) is 1 since r = 0. The printout uses
  S = K = 40, T = 2, σ = 60% so the convergence is visible over several
  barriers; with the exercise 16 data C − V is already 4e-11 at B = 20.
- Exercise 18: Gamma of the knock-out turns negative near the barrier
  (S = 36: Δ = 0.674361, Γ = −0.009534).
- Exercises 1, 15, 19-21 have no numbers; the values above use the stated
  parameters.
- Tests use step sizes chosen so truncation error dominates round-off on any
  IEEE double machine, and tolerances with margin. They were also run with
  clang at -O0 and with `-march=native -ffp-contract=fast` (FMA contraction),
  with the same results. Far out of the money a call value underflows to 0, so
  the "C > 0" check is only strict for S > 0.8K.
