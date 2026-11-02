// Section 7.2: finite difference solutions of ODEs. Convergence of the
// forward, backward and central schemes, stability, and a boundary value
// problem solved through a tridiagonal system.

#include <cmath>
#include <cstdio>
#include <p4mfe/ch07/ode.hpp>

int main() {
  using namespace p4mfe::ch07;

  const auto f = [](double x, double y) { return -2 * x * y; };
  const auto exact = [](double x) { return std::exp(-x * x); };
  std::printf("y' = -2 x y, y(0) = 1 on [0, 2]; max error vs. e^(-x^2)\n");
  std::printf("%6s  %12s  %12s  %12s\n", "n", "forward", "backward", "central");
  for (std::size_t n : {10, 20, 40, 80, 160, 320})
    std::printf("%6zu  %12.3e  %12.3e  %12.3e\n", n,
                MaxError(EulerForward(f, 0, 1, 2, n), exact),
                MaxError(EulerBackward(f, 0, 1, 2, n), exact),
                MaxError(Leapfrog(f, 0, 1, 2, n), exact));

  std::printf("\ny' = -20 y, y(0) = 1: y(3) = %.3e\n", std::exp(-60.0));
  std::printf("%6s  %12s  %12s  %12s\n", "n", "forward", "backward", "central");
  const auto decay = [](double, double y) { return -20 * y; };
  for (std::size_t n : {20, 40, 100, 1000})
    std::printf("%6zu  %12.3e  %12.3e  %12.3e\n", n,
                EulerForward(decay, 0, 1, 3, n).y.back(),
                EulerBackward(decay, 0, 1, 3, n).y.back(),
                Leapfrog(decay, 0, 1, 3, n).y.back());

  std::printf(
      "\ny'' - y = 0, y(0) = 0, y(1) = sinh(1); max error vs. sinh(x)\n");
  std::printf("%6s  %12s  %8s\n", "n", "error", "order");
  double prev = 0;
  for (std::size_t n : {4, 8, 16, 32, 64}) {
    const auto sol =
        SolveBvp([](double) { return 0.0; }, [](double) { return -1.0; },
                 [](double) { return 0.0; }, 0, 1, 0, std::sinh(1.0), n);
    const double e = MaxError(sol, [](double x) { return std::sinh(x); });
    std::printf("%6zu  %12.3e  %8.3f\n", n, e,
                prev > 0 ? std::log2(prev / e) : 0.0);
    prev = e;
  }
}
