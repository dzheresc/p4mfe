# Chapter 9 exercises (section 9.6)

Solutions to exercises 1-3 of Stefanica, *A Primer for the Mathematics of
Financial Engineering*, 2nd ed., section 9.6 (page 293).

Each header in `include/p4mfe/ch09/exercises/` (namespace `p4mfe::ch09::ex`)
has a Python module with the same name in
`python/src/p4mfe/ch09_lagrange_portfolios/exercises/`. Every header starts
with the exercise statement and the full derivation or proof as a comment.
The code reuses chapter 9's `lagrange_stationary_point`,
`classify_constrained` and `lagrangian_gradient` (section 9.1) and
`covariance_matrix`, `frontier_constants`, `min_variance_portfolio`,
`frontier_variance` and `max_return_portfolio` (sections 9.2-9.4), plus
chapter 8's `determinant` in the tests.

```sh
# C++: tests and a program that prints every answer
ctest --test-dir build -R Ch09Ex
./build/cpp/ch09_lagrange_portfolios/exercises/ch09_exercises_solutions

# Python: tests and the same printout
pytest python/tests/ch09_lagrange_portfolios/exercises
python -m p4mfe.ch09_lagrange_portfolios.exercises
```

| # | File | Answer |
|---|------|--------|
| 1 | `ex01_constrained_extrema` | The feasible set is compact and the constraint gradients are independent on it. Lagrange gives x = (−1/(2(λ₁+λ₂)), 1/(λ₁+2λ₂), −1/(2λ₂), −2/λ₁); four stationary points. **max f = 12.761255** at x = (2.137669, −0.683412, 0.406720, −2.009150), λ = (0.995446, −1.229346); **min f = −12.761255** at −x (f is odd, g is even). The other two points, f = ±3.434511, are saddles |
| 2 | `ex02_am_gm_hm` | max ∏xᵢ on {xᵢ ≥ 0, Σxᵢ = S} is inside the simplex; Lagrange gives P/xᵢ = −λ, so all xᵢ = S/n: G ≤ A, equality iff all xᵢ are equal. Applied to 1/xᵢ: H ≤ G |
| 3 | `ex03_five_asset_portfolio` | A = 144.98195, B = 15.053812, C = 1.726477, D = 23.690759. (i) w = (0.067354, 0.217299, −0.002244, 0.406893, 0.310697), **σ = 14.12%** (0.141214). (ii) **μ = 19.92%** (0.199151), w = (−0.395503, 0.096062, 0.045839, 0.709880, 0.543722) |
