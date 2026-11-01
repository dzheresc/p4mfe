# Chapter 5 pseudocode

The pseudocode tables of Chapter 5, transcribed from the book (2nd edition,
pages 136-150). The pseudocode is copied as printed; only the typesetting is
plain text (`x_new` for 𝑥_new, `||·||` for the Euclidean norm, a fraction
`a / b` written on one line). Each table is followed by where it lives in
this repository and how the code differs from the book.

| Table | Algorithm | C++ (`p4mfe::ch05`) | Python (`p4mfe.ch05_newton_bootstrapping`) |
|-------|-----------|---------------------|---------------------------------------------|
| [5.1](#table-51-bisection-method) | Bisection | `bisection` in [`solvers.hpp`](../include/p4mfe/ch05/solvers.hpp) | `bisection` in [`solvers.py`](../../../python/src/p4mfe/ch05_newton_bootstrapping/solvers.py) |
| [5.2](#table-52-newtons-method) | Newton's method | `newton` | `newton` |
| [5.3](#table-53-secant-method) | Secant method | `secant` | `secant` |
| [5.4](#table-54-n-dimensional-newtons-method) | N-dimensional Newton | `newton_nd` in [`newton_nd.hpp`](../include/p4mfe/ch05/newton_nd.hpp) | `newton_nd` in [`newton_nd.py`](../../../python/src/p4mfe/ch05_newton_bootstrapping/newton_nd.py) |
| [5.5](#table-55-n-dimensional-approximate-newtons-method) | N-dimensional approximate Newton | `approximate_newton_nd` | `approximate_newton_nd` |
| [5.6](#table-56-bond-yield) | Bond yield | `bond_yield_newton` in [`bond_yield.hpp`](../include/p4mfe/ch05/bond_yield.hpp); also `ch02::bond_yield` | `bond_yield_newton` in [`bond_yield.py`](../../../python/src/p4mfe/ch05_newton_bootstrapping/bond_yield.py); also `ch02_integration_bonds.bonds.bond_yield` |
| [5.7](#table-57-implied-volatility) | Implied volatility | `implied_vol_newton` in [`implied_vol.hpp`](../include/p4mfe/ch05/implied_vol.hpp); also `ch03::implied_vol_newton` | `implied_vol_newton` in [`implied_vol.py`](../../../python/src/p4mfe/ch05_newton_bootstrapping/implied_vol.py); also `ch03_probability_black_scholes.implied_vol.implied_vol_newton` |

## Differences common to the solvers

The scalar solvers (5.1-5.3) and the N-dimensional ones (5.4-5.5) share a
`Tolerances{approx, consec, max_iter}` argument and return a result that
also records the number of iterations (and, for the scalar solvers, every
iterate). Compared with the book:

- **Default tolerances.** The book suggests `tol_consec = 10^−6` (or
  `tol_int = 10^−6` for bisection) and `tol_approx = 10^−9`. The code's
  defaults are `1e-9` for both. Pass `Tolerances{1e-9, 1e-6}` to get the
  book's values.
- **Iteration cap.** The book's `while` loops have no limit. The code stops
  after `max_iter` iterations (200 by default) and throws
  (`std::runtime_error` / `RuntimeError`).
- **Early exits.** The code also stops when `f(x)` is exactly 0, or when the
  step rounds to 0 so `x` can no longer change. Newton and the secant method
  throw instead of dividing by a zero (or non-finite) derivative or flat
  secant line, and stop on a non-finite iterate.

With the book's tolerances the code reproduces the example after Table 5.3,
`f(x) = x^4 − 5x^2 + 4 − 1/(1 + e^(x^3))`: the secant method from
`x_−1 = −3.01`, `x_0 = −3` converges in 8 iterations and Newton's method from
`x_0 = −3` in 6, both to `−2.074304`, with the same iterates as the book's
table (`−2.503129020`, `−2.309651708`, … and `−2.5`, `−2.211666639`, …).

## Table 5.1: Bisection Method

```
Input:
a, b = left and right endpoints of the initial interval
f(x) = given function
tol_int = largest admissible size of active interval when solution is found
tol_approx = largest admissible value of |f(x)| when solution is found

Output:
x_M = approximate solution for f(x) = 0

x_L = a; x_R = b
while ( max(|f(x_L)|, |f(x_R)|) > tol_approx ) or ( x_R − x_L > tol_int )
   x_M = (x_L + x_R)/2
   if (f(x_L) f(x_M) < 0)
      x_R = x_M              // active interval [x_L, x_M]
   else
      x_L = x_M;             // active interval [x_M, x_R]
   end
end
```

**Code:** `bisection(f, a, b, tol)`, where `tol.consec` plays the role of
`tol_int`. Differences:

- **Stopping test.** The book stops when `|f|` is small at *both* ends of the
  active interval and the interval is shorter than `tol_int`. The code stops
  when `|f(x_M)|` at the new midpoint is at most `tol_approx` and the
  interval that `x_M` halves is at most `tol_int` long. It returns that
  `x_M`.
- **Sign test.** The book tests `f(x_L) f(x_M) < 0`; the code compares the
  signs of `f(x_L)` and `f(x_M)`, which cannot underflow to 0. It keeps
  `f(x_L)` from the previous step instead of evaluating it again, so each
  iteration costs one evaluation of `f`, where the book's pseudocode as
  written evaluates `f` up to four times.
- **Checks.** The code requires `a < b` and `f(a)`, `f(b)` of opposite signs
  (or returns at once if either is 0), and stops when the interval is one
  ulp wide. The book assumes the bracket is valid.

## Table 5.2: Newton's Method

```
Input:
x_0 = initial guess
f(x) = given function
tol_approx = largest admissible value of |f(x)| when solution is found
tol_consec = largest admissible distance between
      two consecutive approximations when solution is found

Output:
x_new = approximate solution for f(x) = 0

x_new = x_0; x_old = x_0 − 1
while ( |f(x_new)| > tol_approx ) or ( |x_new − x_old| > tol_consec )
   x_old = x_new
   x_new = x_old − f(x_old) / f'(x_old)
end
```

**Code:** `newton(f, df, x0, tol)`. Same recursion and stopping test. The
derivative is a second callable `df`, where the book's input list names only
`f(x)`. Instead of seeding `x_old = x_0 − 1` to force the first step, the
code simply always takes one step unless `f(x_0)` is exactly 0. See the
common differences above.

## Table 5.3: Secant Method

```
Input:
x_−1, x_0 = initial guesses
f(x) = given function
tol_approx = largest admissible value of |f(x)| when solution is found
tol_consec = largest admissible distance between
      two consecutive approximations when solution is found

Output:
x_new = approximate solution for f(x) = 0

x_new = x_0; x_old = x_−1
while ( |f(x_new)| > tol_approx ) or ( |x_new − x_old| > tol_consec )
   x_oldest = x_old
   x_old = x_new
   x_new = x_old − f(x_old) (x_old − x_oldest) / (f(x_old) − f(x_oldest))
end
```

**Code:** `secant(f, x_prev, x0, tol)`, with `x_prev` the book's `x_−1`.
Same recursion and stopping test. The code keeps the previous value of `f`,
so each step costs one evaluation; the pseudocode as written evaluates
`f(x_old)` and `f(x_oldest)` again every time.

## Table 5.4: N-dimensional Newton's Method

```
Input:
x_0 = initial guess
F(x) = given function
tol_approx = largest admissible value of ||F(x)|| when solution is found
tol_consec = largest admissible distance between
      two consecutive approximations when solution is found

Output:
x_new = approximate solution for f(x) = 0

x_new = x_0; x_old = x_0 − 1
while ( ||F(x_new)|| > tol_approx ) or ( ||x_new − x_old|| > tol_consec )
   x_old = x_new
   compute DF(x_old)
   x_new = x_old − solve_linear_system(DF(x_old), F(x_old))
end
```

`||·||` is the Euclidean norm, `||v|| = (Σ_{i=1}^N |v_i|^2)^(1/2)`.

**Code:** `newton_nd(F, jacobian, x0, tol)`. The book's `solve_linear_system`
is `solve_linear(A, b)`, Gaussian elimination with partial pivoting written
out in C++; the Python version calls `numpy.linalg.solve`. The code solves
`DF(x) s = −F(x)` and sets `x_new = x_old + s`, which is the book's update
with the sign moved into the right-hand side. The Jacobian is a callable
the caller supplies (`DF` in the book is "compute DF(x_old)"). Same stopping
test with Euclidean norms; see the common differences above.

## Table 5.5: N-dimensional Approximate Newton's Method

```
Input:
x_0 = initial guess
F(x) = given function
h = parameter for the finite difference approximations of
      partial derivatives in the approximate gradient ΔF(x)
tol_approx = largest admissible value of ||F(x)|| when solution is found
tol_consec = largest admissible distance between
      two consecutive approximations when solution is found

Output:
x_new = approximate solution for f(x) = 0

h = tol_consec; x_new = x_0; x_old = x_0 − 1
while ( ||F(x_new)|| > tol_approx ) or ( ||x_new − x_old|| > tol_consec )
   x_old = x_new
   compute ΔF(x_old)            // use parameter h
   x_new = x_old − solve_linear_system(ΔF(x_old), F(x_old))
end
```

The entries of `ΔF(x)` are the forward differences
`Δ_j F_i(x) = (F_i(x + h e_j) − F_i(x)) / h`, with `e_j` the j-th unit
vector.

**Code:** `approximate_newton_nd(F, x0, tol, h = 1e-6, scheme = Forward)`,
which runs `newton_nd` with `finite_difference_jacobian(F, x, h, scheme)` as
the Jacobian. Differences:

- The book sets `h = tol_consec`. The code takes `h` as its own argument,
  defaulting to `1e-6`, which equals the book's `h` when `tol_consec` is the
  book's suggested `10^−6`.
- `scheme = Difference::Central` (Python: `Difference.CENTRAL`) uses the
  central differences `(F_i(x + h e_j) − F_i(x − h e_j)) / (2h)` that the
  book mentions after the table as the more precise option. Forward
  differences are the default, as in the table.

## Table 5.6: Bond yield

Pseudocode for computing a bond yield.

```
Input:
B = the price of the bond
n = number of cash flows
t_cash_flow = cash flow dates (vector of size n)
v_cash_flow = cash flows (vector of size n)
tol = tolerance for declaring convergence of Newton's method

Output:
x_new = the yield of the bond

x_0 = 0.1;                       // initial guess: 10% yield
x_new = x_0; x_old = x_0 − 1; tol = 10^−6
while (|x_new − x_old| > tol)
   x_old = x_new
   x_new = x_old + ( Σ_{i=1}^n v_cash_flow(i) exp(−x_old t_cash_flow(i)) − B )
                   / ( Σ_{i=1}^n t_cash_flow(i) v_cash_flow(i) exp(−x_old t_cash_flow(i)) )
end
```

**Code:** two versions.

- `ch05::bond_yield_newton(cf, price, y0 = 0.1, tol)` runs the generic
  `newton` with `f(y) = bond_price(cf, y) − price` and
  `f'(y) = −dollar_duration(cf, y)`. That is the book's update. It stops on
  both `|f| ≤ 1e-10` and `|step| ≤ 1e-12`
  (`bond_yield_tolerances`), where the book stops on the step alone with
  `tol = 10^−6`. The initial guess is the book's 0.1.
  `bond_yield_secant` and `bond_yield_bisection` solve the same equation
  with the other two methods, for comparison; the book has no table for
  them.
- `ch02::bond_yield(cf, price, tol = 1e-12, y0 = 0.1, max_iter = 100)` is a
  hand-written loop that is the book's table line for line: it computes both
  sums, steps by their ratio, and stops when the step is at most `tol`. Only
  the default tolerance (`1e-12` instead of `10^−6`) and the iteration cap
  differ.

Note that the book's table lists `tol` as an input and then overwrites it
with `10^−6`; both versions in the code use the `tol` they are given.

## Table 5.7: Implied volatility

Pseudocode for computing implied volatility.

```
Input:
C = price of the call option
S = spot price of the underlying asset
K = strike price of the option
T = maturity of the option
q = dividend yield of the underlying asset
r = risk-free constant rate
tol = tolerance for declaring convergence of Newton's method
f_BS(x) = Black–Scholes value of a call; here, x = volatility
vega_BS(x) = vega of the call; x = volatility

Output:
x_new = implied volatility

x_0 = 0.25;                         // initial guess: 25% volatility
x_new = x_0; x_old = x_0 − 1; tol = 10^−6
while (|x_new − x_old| > tol)
   x_old = x_new
   x_new = x_new − (f_BS(x_new) − C) / vega_BS(x_new)
end
```

(The update is printed with `x_new` on the right-hand side; after
`x_old = x_new` it is the Newton step from `x_old`.)

**Code:** two versions.

- `ch05::implied_vol_newton(type, price, S, K, T, r, q, tol)` runs the
  generic `newton` with `f(σ) = bs_price(...) − price` and `f'(σ) = vega`.
- `ch03::implied_vol_newton(type, price, S, K, T, r, q, tol = 1e-10)` is a
  hand-written loop closer to the table.

Differences from the book, for both:

- **Initial guess.** The book starts at 25% volatility. The code starts at
  `σ_0 = sqrt(2 |ln(S/K) + (r − q) T| / T)` (`implied_vol_initial_guess`,
  floored at `1e-4`), the inflection point of the price as a function of
  σ, from which Newton's method converges monotonically.
- **Calls and puts.** The book prices a call; the code takes an
  `OptionType` and works for puts too (vega is the same).
- **Price check.** The code first checks that the price is inside the
  no-arbitrage bounds and throws otherwise. The book assumes it is.
- **Stopping.** The book stops when consecutive iterates are within
  `10^−6`. `ch03::implied_vol_newton` stops when they are within `tol`
  (`1e-10` by default) or when the price is matched to rounding error,
  which handles deep in- or out-of-the-money options where vega is tiny.
  `ch05::implied_vol_newton` uses the generic test, `|f| ≤ 1e-9` and
  `|step| ≤ 1e-9`. Both throw if Newton does not converge. If an iterate
  leaves `(0, ∞)`, `ch03::implied_vol_newton` throws with that message, and
  `ch05::implied_vol_newton` throws from `bs_price`, which rejects
  `σ ≤ 0`.
- `f_BS` and `vega_BS` are `bs_price` and `bs_greeks(...).vega` from
  chapter 3, which use the exact `N(x)` (see Table 3.2).
