// Section 5.1: bisection, Newton and secant on
//   f(x) = x^4 - 5x^2 + 4 - 1/(1 + e^(x^3)),
// a quartic with four real roots, perturbed. Prints the iterates and the
// observed order of convergence of Newton and secant for the root near
// 0.95. (Bisection errors do not even decrease monotonically, so no order is
// shown for it; on average they halve each step.)

#include <cmath>
#include <cstdio>
#include <p4mfe/ch05/solvers.hpp>

int main() {
  using namespace p4mfe::ch05;
  const auto f = [](double x) {
    return std::pow(x, 4) - 5 * x * x + 4 - 1 / (1 + std::exp(std::pow(x, 3)));
  };
  const auto df = [](double x) {
    const double e = std::exp(std::pow(x, 3));
    return 4 * std::pow(x, 3) - 10 * x + 3 * x * x * e / ((1 + e) * (1 + e));
  };
  const Tolerances tol{1e-14, 1e-14};
  const double root = newton(f, df, 0.5, {1e-16, 1e-16}).x;

  const auto show = [&](const char* name, const RootResult& r, bool order) {
    std::printf("\n%s: %zu iterations, x = %.15f\n", name, r.iterations, r.x);
    std::printf("%4s  %20s  %12s  %8s\n", "k", "x_k", "|x_k - x*|", "order");
    double e2 = NAN, e1 = NAN;
    for (std::size_t k = 0; k < r.iterates.size(); ++k) {
      const double e = std::abs(r.iterates[k] - root);
      const double p = (order && k >= 2 && e > 0 && e1 > 0 && e2 > 0)
                           ? convergence_order(e2, e1, e)
                           : NAN;
      if (r.iterates.size() <= 16 || k < 6 || k + 3 >= r.iterates.size())
        std::printf("%4zu  %20.15f  %12.3e  %8.3f\n", k, r.iterates[k], e, p);
      else if (k == 6)
        std::printf("%4s\n", "...");
      e2 = e1;
      e1 = e;
    }
  };
  show("Bisection on [0.5, 1]", bisection(f, 0.5, 1.0, tol), false);
  show("Newton from 0.5", newton(f, df, 0.5, tol), true);
  show("Secant from 0.49, 0.5", secant(f, 0.49, 0.5, tol), true);

  std::printf("\nOther roots: ");
  for (double x0 : {-2.5, -0.5, 2.5})
    std::printf("%.12f  ", newton(f, df, x0, tol).x);
  std::printf("\n");
}
