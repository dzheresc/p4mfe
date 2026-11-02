#pragma once
// Chapter 5, section 5.1: numerical methods for nonlinear problems f(x) = 0.
//
// Three generic solvers, each a template over the callable f:
//   5.1.1 bisection  needs a bracket [a, b] with f(a) f(b) < 0; always
//                    converges, linearly (the interval halves each step).
//   5.1.2 Newton     x_{k+1} = x_k - f(x_k) / f'(x_k); needs f'; converges
//                    quadratically from a good initial guess.
//   5.1.3 secant     Newton with f' replaced by the slope through the last two
//                    iterates; derivative-free, order (1 + sqrt 5)/2 ~ 1.618.
//
// Stopping follows the book: iterate until BOTH
//   |f(x_k)| <= tol.approx           (the approximate solution solves f = 0)
//   |x_k - x_{k-1}| <= tol.consec    (consecutive iterates agree;
//                                     for bisection: the bracket is that short)
// hold. An iteration that can no longer move x (f(x) == 0 exactly, or the
// step rounds to zero) also stops. Running out of max_iter throws.

#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch05 {

struct Tolerances {
  double approx = 1e-9;  // tol_approx: on |f(x)|
  double consec =
      1e-9;  // tol_consec: on |x_k - x_{k-1}| (tol_int for bisection)
  std::size_t max_iter = 200;
};

struct RootResult {
  double x;                      // the approximate root
  std::size_t iterations;        // number of updates of x
  std::vector<double> iterates;  // x_0, x_1, ..., x (for convergence studies)
};

// Initial capacity of RootResult::iterates. Newton and the secant method
// usually converge in under 16 steps; bisection needs about 50 to reach
// double precision.
inline constexpr std::size_t kFastIteratesHint = 16;
inline constexpr std::size_t kBisectionIteratesHint = 64;

template <class F>
concept ScalarFunction =
    std::invocable<F, double> &&
    std::convertible_to<std::invoke_result_t<F, double>, double>;

// --------------------------------------------------------------- 5.1.1 -----
// Keeps a bracket [x_L, x_R] with a sign change and replaces one end by the
// midpoint x_M each step. Returns the last x_M; `iterates` holds the
// midpoints and `iterations` counts them.
template <ScalarFunction F>
[[nodiscard]] RootResult bisection(F&& f, double a, double b,
                                   const Tolerances& tol = {}) {
  if (!(a < b)) throw std::invalid_argument("bisection: need a < b");
  double xl = a, xr = b;
  double fl = f(xl), fr = f(xr);
  if (fl == 0.0) return {xl, 0, {xl}};
  if (fr == 0.0) return {xr, 0, {xr}};
  if ((fl < 0.0) == (fr < 0.0))
    throw std::invalid_argument(
        "bisection: f(a) and f(b) must have opposite signs");
  RootResult res{0.5 * (xl + xr), 0, {}};
  // About log2((b - a) / tol) iterates; one allocation covers typical use.
  res.iterates.reserve(kBisectionIteratesHint);
  while (res.iterations < tol.max_iter) {
    const double xm = 0.5 * (xl + xr);
    const double fm = f(xm);
    res.x = xm;
    res.iterates.push_back(xm);
    ++res.iterations;
    if (fm == 0.0 || (std::abs(fm) <= tol.approx && xr - xl <= tol.consec))
      return res;
    if (xm <= xl || xm >= xr) return res;  // bracket is one ulp wide
    if ((fm < 0.0) == (fl < 0.0)) {
      xl = xm;
      fl = fm;
    } else {
      xr = xm;
    }
  }
  throw std::runtime_error("bisection: no convergence within max_iter");
}

// --------------------------------------------------------------- 5.1.2 -----
template <ScalarFunction F, ScalarFunction DF>
[[nodiscard]] RootResult newton(F&& f, DF&& df, double x0,
                                const Tolerances& tol = {}) {
  RootResult res{x0, 0, {}};
  res.iterates.reserve(kFastIteratesHint);  // one allocation for typical use
  res.iterates.push_back(x0);
  double fx = f(x0);
  while (res.iterations < tol.max_iter) {
    if (fx == 0.0) return res;
    const double d = df(res.x);
    if (d == 0.0 || !std::isfinite(d))
      throw std::runtime_error("newton: f'(x) is zero or not finite");
    const double next = res.x - fx / d;
    if (!std::isfinite(next))
      throw std::runtime_error("newton: iterate is not finite");
    const double step = std::abs(next - res.x);
    res.x = next;
    res.iterates.push_back(next);
    ++res.iterations;
    fx = f(next);
    if ((std::abs(fx) <= tol.approx && step <= tol.consec) || step == 0.0)
      return res;
  }
  throw std::runtime_error("newton: no convergence within max_iter");
}

// --------------------------------------------------------------- 5.1.3 -----
// Needs two starting points x_{-1} and x_0; the book takes x_{-1} close to
// x_0. Only one new evaluation of f per step, which is why the secant method
// often beats Newton per unit of work when f' is as expensive as f.
template <ScalarFunction F>
[[nodiscard]] RootResult secant(F&& f, double x_prev, double x0,
                                const Tolerances& tol = {}) {
  RootResult res{x0, 0, {}};
  res.iterates.reserve(kFastIteratesHint);  // one allocation for typical use
  res.iterates.push_back(x0);
  double f_prev = f(x_prev);
  double fx = f(x0);
  while (res.iterations < tol.max_iter) {
    if (fx == 0.0) return res;
    if (fx == f_prev) throw std::runtime_error("secant: flat secant line");
    const double next = res.x - fx * (res.x - x_prev) / (fx - f_prev);
    if (!std::isfinite(next))
      throw std::runtime_error("secant: iterate is not finite");
    const double step = std::abs(next - res.x);
    x_prev = res.x;
    f_prev = fx;
    res.x = next;
    res.iterates.push_back(next);
    ++res.iterations;
    fx = f(next);
    if ((std::abs(fx) <= tol.approx && step <= tol.consec) || step == 0.0)
      return res;
  }
  throw std::runtime_error("secant: no convergence within max_iter");
}

// Estimated order of convergence from the last three errors e_k = |x_k - x*|:
//   p ~ ln(e_{k+1} / e_k) / ln(e_k / e_{k-1}).
// About 1 for bisection (on average), 2 for Newton, 1.618 for secant.
[[nodiscard]] inline double convergence_order(double e_prev, double e,
                                              double e_next) {
  return std::log(e_next / e) / std::log(e / e_prev);
}

}  // namespace p4mfe::ch05
