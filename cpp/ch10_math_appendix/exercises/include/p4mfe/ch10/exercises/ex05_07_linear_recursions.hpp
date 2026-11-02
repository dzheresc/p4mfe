#pragma once
// Exercises 5-7 (section 10.10): closed forms of linear recursions.

#include <array>
#include <cmath>
#include <functional>
#include <p4mfe/ch05/newton_nd.hpp>
#include <p4mfe/ch10/interpolation.hpp>
#include <p4mfe/ch10/linear_recursions.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch10::ex {

// Fits x_n = sum_j c_j b_j(n) to the first terms x_0, ..., x_{k-1} (k basis
// functions): solves the k x k system b_j(i) c_j = x_i with chapter 5's
// solve_linear.
using Basis = std::vector<std::function<double(unsigned)>>;

[[nodiscard]] inline Vector ExFitConstants(const Basis& basis,
                                           const Vector& init) {
  const std::size_t k = basis.size();
  if (init.size() != k)
    throw std::invalid_argument(
        "ex_fit_constants: need one initial term per basis function");
  ch05::Matrix matrix(k, k);
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j)
      matrix(i, j) = basis[j](static_cast<unsigned>(i));
  return ch05::SolveLinear(std::move(matrix), init);
}

[[nodiscard]] inline double ExEvaluate(const Basis& basis, const Vector& c,
                                       unsigned n) {
  double s = 0.0;
  for (std::size_t j = 0; j < basis.size(); ++j) s += c[j] * basis[j](n);
  return s;
}

// ------------------------------------------------------------ Exercise 5 ---
// Find the general form of x_{n+3} = 2 x_{n+1} + x_n, n >= 0, with x_0 = 1,
// x_1 = -1, x_2 = 1.
//
// Characteristic polynomial: z^3 - 2z - 1 = (z + 1)(z^2 - z - 1), with roots
//   -1,  phi = (1 + sqrt 5)/2,  psi = (1 - sqrt 5)/2.
// The roots are distinct, so the general solution is
//   x_n = c_1 (-1)^n + c_2 phi^n + c_3 psi^n.
// The initial conditions give
//   c_1 + c_2 + c_3 = 1,  -c_1 + c_2 phi + c_3 psi = -1,  c_1 + c_2 phi^2 + c_3
//   psi^2 = 1.
// Since phi^2 = phi + 1 and psi^2 = psi + 1, the third minus the first is
// c_2 phi + c_3 psi = 0; then the second gives c_1 = 1, and c_2 + c_3 = 0
// with c_2 (phi - psi) = 0 gives c_2 = c_3 = 0. So
//   x_n = (-1)^n  for all n >= 0.
// Check: x_3 = 2 x_1 + x_0 = -1, x_4 = 2 x_2 + x_1 = 1.
[[nodiscard]] inline Basis Ex05Basis() {
  const double phi = 0.5 * (1.0 + std::sqrt(5.0)),
               psi = 0.5 * (1.0 - std::sqrt(5.0));
  return {[](unsigned n) { return n % 2 == 0 ? 1.0 : -1.0; },
          [phi](unsigned n) { return std::pow(phi, static_cast<double>(n)); },
          [psi](unsigned n) { return std::pow(psi, static_cast<double>(n)); }};
}
inline constexpr std::array<double, 3> kEx05Recursion{
    1.0, 2.0, 0.0};  // x_{n+3} = 1 x_n + 2 x_{n+1} + 0 x_{n+2}
inline constexpr std::array<double, 3> kEx05Init{1.0, -1.0, 1.0};
[[nodiscard]] inline Vector Ex05Constants() {
  return ExFitConstants(Ex05Basis(), {1.0, -1.0, 1.0});
}
[[nodiscard]] inline double Ex05ClosedForm(unsigned n) noexcept {
  return n % 2 == 0 ? 1.0 : -1.0;
}

// ------------------------------------------------------------ Exercise 6 ---
// x_{n+1} = 3 x_n + 2, x_0 = 1.
//
// (i) Writing the recursion for n + 1, x_{n+2} = 3 x_{n+1} + 2, and
// subtracting: x_{n+2} - x_{n+1} = 3 x_{n+1} - 3 x_n, i.e.
//   x_{n+2} = 4 x_{n+1} - 3 x_n,  x_0 = 1, x_1 = 3 + 2 = 5.
//
// (ii) z^2 - 4z + 3 = (z - 1)(z - 3), so x_n = c_1 + c_2 3^n with
// c_1 + c_2 = 1 and c_1 + 3 c_2 = 5: c_2 = 2, c_1 = -1, and
//   x_n = 2 * 3^n - 1.
// (Equivalently, x_n + 1 = 3 (x_{n-1} + 1) = ... = 3^n (x_0 + 1).)
[[nodiscard]] inline double Ex06ClosedForm(unsigned n) {
  return 2.0 * std::pow(3.0, static_cast<double>(n)) - 1.0;
}
// The same from chapter 10's second order solver and first order formula.
[[nodiscard]] inline SecondOrderRecursion Ex06SecondOrder() {
  return SolveSecondOrder(4.0, -3.0, 1.0, 5.0);
}
[[nodiscard]] inline double Ex06FirstOrder(unsigned n) noexcept {
  return FirstOrderAffine(3.0, 2.0, 1.0, n);
}

// ------------------------------------------------------------ Exercise 7 ---
// x_{n+1} = 3 x_n + n + 2, x_0 = 1.
//
// (i) For n + 1: x_{n+2} = 3 x_{n+1} + n + 3. Subtracting,
//   x_{n+2} = 4 x_{n+1} - 3 x_n + 1.
// The constant 1 goes away after one more shift and subtraction:
//   x_{n+3} - x_{n+2} = 4 x_{n+2} - 4 x_{n+1} - 3 x_{n+1} + 3 x_n, i.e.
//   x_{n+3} = 5 x_{n+2} - 7 x_{n+1} + 3 x_n,
// with x_0 = 1, x_1 = 3 + 0 + 2 = 5, x_2 = 15 + 1 + 2 = 18.
//
// (ii) z^3 - 5z^2 + 7z - 3 = (z - 1)^2 (z - 3): a double root 1 and a simple
// root 3, so x_n = c_1 + c_2 n + c_3 3^n. From x_0, x_1, x_2:
//   c_1 + c_3 = 1,  c_1 + c_2 + 3 c_3 = 5,  c_1 + 2 c_2 + 9 c_3 = 18.
// Subtracting consecutive equations: c_2 + 2 c_3 = 4, c_2 + 6 c_3 = 13, so
// c_3 = 9/4, c_2 = -1/2, c_1 = -5/4:
//   x_n = (3^(n+2) - 2n - 5) / 4.
// (Check against the original recursion: the particular solution -n/2 - 5/4
// satisfies p(n+1) = 3 p(n) + n + 2, and 9/4 * 3^n solves the homogeneous
// part.)
[[nodiscard]] inline Basis Ex07Basis() {
  return {[](unsigned) { return 1.0; },
          [](unsigned n) { return static_cast<double>(n); },
          [](unsigned n) { return std::pow(3.0, static_cast<double>(n)); }};
}
inline constexpr std::array<double, 3> kEx07Recursion{
    3.0, -7.0, 5.0};  // x_{n+3} = 3 x_n - 7 x_{n+1} + 5 x_{n+2}
inline constexpr std::array<double, 3> kEx07Init{1.0, 5.0, 18.0};
[[nodiscard]] inline Vector Ex07Constants() {
  return ExFitConstants(Ex07Basis(), {1.0, 5.0, 18.0});
}
[[nodiscard]] inline double Ex07ClosedForm(unsigned n) {
  return (std::pow(3.0, static_cast<double>(n) + 2.0) - 2.0 * n - 5.0) / 4.0;
}
// x_0..x_{count-1} from the original first order recursion.
[[nodiscard]] inline std::vector<double> Ex07Direct(std::size_t count) {
  std::vector<double> x{1.0};
  for (std::size_t n = 0; x.size() < count; ++n)
    x.push_back(3.0 * x[n] + static_cast<double>(n) + 2.0);
  x.resize(count);
  return x;
}

}  // namespace p4mfe::ch10::ex
