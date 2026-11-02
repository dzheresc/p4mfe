// Section 2.4.2: integral_0^2 e^(-x^2) dx with the midpoint, trapezoidal and
// Simpson's rules for n = 4, 8, ..., showing O(h^2) vs O(h^4) convergence,
// then each rule run to a tolerance of 1e-7 (section 2.4.1).

#include <cmath>
#include <cstdio>
#include <numbers>
#include <p4mfe/ch02/integration.hpp>

int main() {
  using namespace p4mfe::ch02;
  const auto f = [](double x) { return std::exp(-x * x); };
  const double exact = 0.5 * std::sqrt(std::numbers::pi) * std::erf(2.0);

  std::printf("exact = %.12f\n\n", exact);
  std::printf("%6s  %14s  %14s  %14s\n", "n", "midpoint err", "trapezoid err",
              "simpson err");
  for (std::size_t n = 4; n <= 512; n *= 2) {
    std::printf(
        "%6zu  %14.3e  %14.3e  %14.3e\n", n, MidpointRule(f, 0, 2, n) - exact,
        TrapezoidalRule(f, 0, 2, n) - exact, SimpsonRule(f, 0, 2, n) - exact);
  }

  std::printf("\nTolerance 1e-7:\n");
  const struct {
    const char* name;
    Rule rule;
  } rules[] = {{"midpoint", Rule::kMidpoint},
               {"trapezoidal", Rule::kTrapezoidal},
               {"simpson", Rule::kSimpson}};
  for (const auto& [name, rule] : rules) {
    const auto res = IntegrateToTolerance(f, 0, 2, 1e-7, rule);
    std::printf("  %-12s n = %6zu  value = %.12f\n", name, res.intervals,
                res.value);
  }
}
