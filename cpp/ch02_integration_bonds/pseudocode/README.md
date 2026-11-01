# Chapter 2 pseudocode

The pseudocode tables of Chapter 2, transcribed from the book (2nd edition,
pages 46-48 and 66-69). The pseudocode is copied as printed; only the
typesetting is plain text (`∫_a^b f(x)` for the integral sign, `^2` for a
superscript). Each table is followed by where it lives in this repository
and how the code differs from the book.

| Table | Algorithm | C++ (`p4mfe::ch02`) | Python (`p4mfe.ch02_integration_bonds`) |
|-------|-----------|---------------------|------------------------------------------|
| [2.1](#table-21-midpoint-rule) | Midpoint rule | `midpoint_rule` in [`integration.hpp`](../include/p4mfe/ch02/integration.hpp) | `midpoint_rule` in [`integration.py`](../../../python/src/p4mfe/ch02_integration_bonds/integration.py) |
| [2.2](#table-22-trapezoidal-rule) | Trapezoidal rule | `trapezoidal_rule` | `trapezoidal_rule` |
| [2.3](#table-23-simpsons-rule) | Simpson's rule | `simpson_rule` | `simpson_rule` |
| [2.4](#table-24-integral-with-given-tolerance) | Integral with given tolerance | `integrate_to_tolerance` | `integrate_to_tolerance` |
| [2.5](#table-25-bond-price-given-the-zero-rate-curve) | Bond price from the zero rate curve | `bond_price_from_zero_rate` in [`bonds.hpp`](../include/p4mfe/ch02/bonds.hpp) | `bond_price_from_zero_rate` in [`bonds.py`](../../../python/src/p4mfe/ch02_integration_bonds/bonds.py) |
| [2.6](#table-26-bond-price-given-the-instantaneous-interest-rate-curve) | Bond price from the instantaneous rate curve | `bond_price_from_inst_rate` | `bond_price_from_inst_rate` |
| [2.7](#table-27-price-duration-and-convexity-given-the-yield) | Price, duration and convexity from the yield | `bond_metrics` | `bond_metrics` |

Differences common to all tables: the book indexes from 1 (`for i = 1 : n`),
the code from 0; the book passes the integrand as a routine `f_int(x)`, the
code takes any callable `f` (in Python, `f` must accept a NumPy array, since
the nodes are evaluated in one call); and the book's separate `n`,
`t_cash_flow`, `v_cash_flow` inputs are one `CashFlows` value (`times`,
`amounts`) in the code.

## Table 2.1: Midpoint Rule

```
Input:
a = left endpoint of the integration interval
b = right endpoint of the integration interval
n = number of partition intervals
f_int(x) = routine evaluating f(x)

Output:
I_midpoint = Midpoint Rule approximation of ∫_a^b f(x)

h = (b − a)/n;  I_midpoint = 0
for i = 1 : n
   I_midpoint = I_midpoint + f_int(a + (i − 1/2)h)
end
I_midpoint = h · I_midpoint
```

**Code:** `midpoint_rule(f, a, b, n)`. Same algorithm. The C++ loop runs
`i = 0 .. n-1` and evaluates `f(a + (i + 0.5) h)`, which is the book's
`i − 1/2` shifted by one. Both versions reject `n = 0`.

## Table 2.2: Trapezoidal Rule

```
Input:
a = left endpoint of the integration interval
b = right endpoint of the integration interval
n = number of partition intervals
f_int(x) = routine evaluating f(x)

Output:
I_trap = Trapezoidal Rule approximation of ∫_a^b f(x)

h = (b − a)/n
I_trap = f_int(a)/2 + f_int(b)/2
for i = 1 : (n − 1)
   I_trap = I_trap + f_int(a + ih)
end
I_trap = h · I_trap
```

**Code:** `trapezoidal_rule(f, a, b, n)`. The C++ version is line for line
the book's. The Python version evaluates all `n + 1` nodes at once and
subtracts half of the two end values from their sum, which is the same
quantity.

## Table 2.3: Simpson's Rule

```
Input:
a = left endpoint of the integration interval
b = right endpoint of the integration interval
n = number of partition intervals
f_int(x) = routine evaluating f(x)

Output:
I_simpson = Simpson's Rule approximation of ∫_a^b f(x)

h = (b − a)/n
I_simpson = f_int(a)/6 + f_int(b)/6
for i = 1 : (n − 1)
   I_simpson = I_simpson + f_int(a + ih)/3
end
for i = 1 : n
   I_simpson = I_simpson + 2 f_int(a + (i − 1/2)h)/3
end
I_simpson = h · I_simpson
```

**Code:** `simpson_rule(f, a, b, n)`. Same nodes and weights: the code sums
the end points, the interior nodes and the midpoints separately and returns
`h/6 · (ends + 2·nodes + 4·mids)`, which is the book's `1/6`, `1/3`, `2/3`
weights with `1/6` factored out. As in the book, `n` counts intervals, not
nodes, and each interval contributes its midpoint, so there is no
"`n` must be even" restriction.

## Table 2.4: Integral with given tolerance

Pseudocode for computing an approximate value of an integral with given
tolerance.

```
Input:
tol = prescribed tolerance
I_numerical(n) = result of the numerical integration rule with
      n intervals; any integration rule can be used

Output:
I_approx = approximation of ∫_a^b f(x) with tolerance tol

n = 4;  I_old = I_numerical(n)            // 4 intervals initial partition
n = 2n; I_new = I_numerical(n)
while (abs(I_new − I_old) > tol)
   I_old = I_new
   n = 2n
   I_new = I_numerical(n)
end
I_approx = I_new
```

**Code:** `integrate_to_tolerance(f, a, b, tol, rule = Simpson, n0 = 4,
max_n = 2^24)`, built on `detail::refine` (Python: `_refine`). Differences:

- The rule is chosen by the `Rule` enum (`Midpoint`, `Trapezoidal`,
  `Simpson`); Simpson is the default. The starting `n` is a parameter,
  defaulting to the book's 4.
- The loop stops when `|I_new − I_old| < tol`, where the book stops at
  `≤ tol`. This only matters when the difference is exactly `tol`.
- There is a cap: if `n` would exceed `max_n` without meeting the tolerance,
  the function throws (`std::runtime_error` / `RuntimeError`) instead of
  looping forever. A non-positive `tol` is rejected.
- It returns `IntegrationResult{value, intervals}`, so the caller also gets
  the final `n`.
- Like the book's pseudocode, each `I_numerical(2n)` is computed from scratch.
  The node reuse the book describes after the table (for the trapezoidal and
  Simpson rules, half of the nodes of the `2n` partition are already known)
  is not implemented.

## Table 2.5: Bond price given the zero rate curve

Pseudocode for computing the bond price given the zero rate curve.

```
Input:
n = number of cash flows
t_cash_flow = vector of cash flow dates (of size n)
v_cash_flow = vector of cash flows (of size n)
r_zero(t) = zero rate corresponding to time t

Output:
B = bond price

B = 0
for i = 1 : n
   disc(i) = exp(−t_cash_flow(i) r_zero(t_cash_flow(i)) )
   B = B + v_cash_flow(i) disc(i)
end
```

**Code:** `bond_price_from_zero_rate(cf, zero)`. It passes the discount
function `t ↦ exp(−zero(t) t)` to `bond_price_from_discount(cf, disc)`, which
runs the book's loop. The discount factors are not kept in a `disc` vector.

## Table 2.6: Bond price given the instantaneous interest rate curve

Pseudocode for computing the bond price given the instantaneous interest
rate curve.

```
Input:
n = number of cash flows
t_cash_flow = vector of cash flow dates (of size n)
v_cash_flow = vector of cash flows (of size n)
r_inst(t) = instantaneous interest rate at time t
tol = vector of tolerances in the numerical approximation
      of discount factor integrals (of size n)

Output:
B = bond price

B = 0
for i = 1 : n
   I_numerical(i) = result of the numerical integration of r_inst(t)
         on the interval [0, t_cash_flow(i)] with tolerance tol(i);
   disc(i) = exp(−I_numerical(i))
   B = B + v_cash_flow(i) disc(i)
end
```

**Code:** `bond_price_from_inst_rate(cf, r, tol = 1e-12)`. Two differences:

- The book integrates `r_inst` over `[0, t_cash_flow(i)]` afresh for every
  cash flow. The code integrates only the new piece `[t_{i−1}, t_i]` and adds
  it to a running total, so the interval `[0, t_1]` is integrated once rather
  than `n` times. Mathematically the result is the same, but only if the
  cash flow dates are in increasing order. That is not checked: `CashFlows`
  validates only that times and amounts are non-empty and the same size, and
  a date earlier than the one before it silently reuses the previous
  discount factor. The book's version has no such requirement.
- `tol` is one number, not a vector of size `n`, and it applies to each piece
  `[t_{i−1}, t_i]`, so the error in the integral over `[0, t_i]` can add up to
  about `i · tol`. Each piece uses `integrate_to_tolerance` with Simpson's
  rule.

## Table 2.7: Price, duration and convexity given the yield

Pseudocode for computing the price, duration and convexity of a bond given
the yield of the bond.

```
T = bond maturity
n = number of cash flows
t_cash_flow = vector of cash flow dates (of size n)
v_cash_flow = vector of cash flows (of size n)
y = yield of the bond

Output:
B = price of the bond
D = duration of the bond
C = convexity of the bond

B = 0; D = 0; C = 0
for i = 1 : n
   disc(i) = exp(−t_cash_flow(i) y )
   B = B + v_cash_flow(i) disc(i)
   D = D + t_cash_flow(i) v_cash_flow(i) disc(i)
   C = C + t_cash_flow(i)^2 v_cash_flow(i) disc(i)
end
D = D/B; C = C/B
```

**Code:** `bond_metrics(cf, y)` returns `BondMetrics{price, duration,
convexity}`; `bond_price`, `bond_duration` and `bond_convexity` return one of
the three. Same algorithm. The book lists the maturity `T` as an input but
never uses it; the code has no such parameter.

The book's example after the table (semiannual 6% coupon, 20 months,
yield 6.5%, so `t_cash_flow = [2/12 8/12 14/12 20/12]`,
`v_cash_flow = [3 3 3 103]`) gives `B = 101.046193`; `bond_metrics` returns
`101.0461929`, duration `1.5804215`, convexity `2.5916859`.
