// Prints the numerical answers and checks for the exercises of section 7.6.
//
//   ./build/cpp/ch07_finite_differences_bs_pde/exercises/ch07_exercises_solutions

#include <p4mfe/ch07/exercises/ex01_02_difference_orders.hpp>
#include <p4mfe/ch07/exercises/ex03_04_stencils.hpp>
#include <p4mfe/ch07/exercises/ex05_nonsymmetric_stencils.hpp>
#include <p4mfe/ch07/exercises/ex06_butterfly_spread.hpp>
#include <p4mfe/ch07/exercises/ex07_bull_spread.hpp>
#include <p4mfe/ch07/exercises/ex08_butterfly_delta.hpp>
#include <p4mfe/ch07/exercises/ex09_euler_convergence.hpp>
#include <p4mfe/ch07/exercises/ex10_bvp_linear_system.hpp>
#include <p4mfe/ch07/exercises/ex11_first_order_system.hpp>
#include <p4mfe/ch07/exercises/ex12_gamma_theta.hpp>
#include <p4mfe/ch07/exercises/ex13_14_bs_pde_solutions.hpp>

#include <cmath>
#include <cstdio>

using namespace p4mfe;
using namespace p4mfe::ch07;
using namespace p4mfe::ch07::ex;

namespace {
void header(const char* title) { std::printf("\n== %s ==\n", title); }

template <class E>
void order_table(const char* label, E&& err, double h0) {
    std::printf("  %s\n", label);
    for (double h = h0; h > h0 / 9.0; h /= 2.0)
        std::printf("    h = %-8.4g error = %-13.6e observed order %.3f\n", h, err(h), ex_observed_order(err, h));
}
}  // namespace

int main() {
    header("Ex 1: central difference for f'(0), f = x^3 e^x - 6 e^x");
    order_table("error ~ 0.45 h^4", ex01_error, 0.2);

    header("Ex 2: forward difference for N'(0)");
    order_table("error ~ -phi(0)/6 h^2 = -0.066490 h^2", ex02_error, 0.2);

    const auto ex = [](double x) { return std::exp(x); };
    header("Ex 3: (-3f(a) + 4f(a+h) - f(a+2h))/(2h), f = e^x, a = 0.3");
    order_table("error ~ -h^2 f'''(a)/3", [&](double h) { return ex03_one_sided(ex, 0.3, h) - std::exp(0.3); }, 0.2);

    header("Ex 4: (f(a-2h) - 4f(a-h) + 6f(a) - 4f(a+h) + f(a+2h))/h^4, f = sin, a = 0.7");
    order_table("error ~ h^2 f^(6)(a)/6",
                [](double h) { return ex04_fourth_derivative([](double x) { return std::sin(x); }, 0.7, h) - std::sin(0.7); },
                0.4);

    header("Ex 5: points a - h, a, a + 2h, f = e^x, a = 0.4");
    order_table("(i)  f'(a)  ~ (3f(a) - 4f(a-h) + f(a+2h))/(6h), error h^2/3 f'''",
                [&](double h) { return ex05_first_derivative_h(ex, 0.4, h) - std::exp(0.4); }, 0.2);
    order_table("(ii) f''(a) ~ (2f(a-h) - 3f(a) + f(a+2h))/(3h^2), error h/3 f'''",
                [&](double h) { return ex05_second_derivative_h(ex, 0.4, h) - std::exp(0.4); }, 0.2);

    const SpreadMarket m;
    header("Ex 6: 1/x^2 butterflies -> C_KK (S = 52, K = 50, T = 0.5, sigma = 25%, r = 4%, q = 1%)");
    std::printf("  C_KK = e^(-rT) phi(d2)/(K sigma sqrt(T)) = %.10f\n", ex06_call_kk(m));
    order_table("butterfly/x^2 - C_KK", [&](double x) { return ex06_scaled_butterfly(x, m) - ex06_call_kk(m); }, 2.0);

    header("Ex 7: 1/x bull spreads -> -C_K = e^(-rT) N(d2)");
    std::printf("  cash-or-nothing call = %.10f\n", ex07_cash_or_nothing(m));
    order_table("bull/x - e^(-rT) N(d2)", [&](double x) { return ex07_scaled_bull(x, m) - ex07_cash_or_nothing(m); },
                2.0);

    header("Ex 8: int f(S) g(S) dS -> g(K), g(S) = cos(S/7), K = 50");
    const auto g = [](double S) { return std::cos(S / 7.0); };
    for (double x : {2.0, 1.0, 0.5, 0.1})
        std::printf("    x = %-5.3g integral = %.12f  g(K) = %.12f  diff/(x^2/12 g''(K)) = %.6f\n", x,
                    ex08_integral(g, 50.0, x), g(50.0),
                    (ex08_integral(g, 50.0, x) - g(50.0)) / (x * x / 12.0 * -std::cos(50.0 / 7.0) / 49.0));

    header("Ex 9: forward Euler for y' = y, y(0) = 1");
    for (std::size_t n : {10u, 100u, 1000u, 10000u})
        std::printf("    n = %-6zu max|e_i| = %.6e  e/(2n) = %.6e  n max|e_i| = %.6f (-> e/2 = %.6f)\n", n,
                    ex09_max_error(n), ex09_bound(n), static_cast<double>(n) * ex09_max_error(n),
                    std::numbers::e / 2.0);

    header("Ex 10: 3x^2 y'' - x y' + y = 0, y(0) = 1, y(1) = 1/2, n = 6");
    const auto s10 = ex10_system(6);
    for (std::size_t k = 0; k < 5; ++k)
        std::printf("    row %zu: %7.2f %7.2f %7.2f | b = %7.3f\n", k + 1, s10.lower[k], s10.diag[k], s10.upper[k],
                    s10.rhs[k]);
    const auto y10 = ex10_solve(6);
    std::printf("  Y = (%.6f, %.6f, %.6f, %.6f, %.6f)\n", y10.y[1], y10.y[2], y10.y[3], y10.y[4], y10.y[5]);
    std::printf("  y(1/2) for n = 6, 12, 24, 48, 96 (no limit: the BVP has no solution):");
    for (std::size_t n : {6u, 12u, 24u, 48u, 96u}) std::printf(" %.4f", ex10_solve(n).y[n / 2]);
    std::printf("\n");

    header("Ex 11: y'' - 2y' + x^2 y = 0 as Y' = f(x, Y), Y(0) = (1, 0)");
    const auto ref = ex11_rk4({1.0, 0.0}, 0.0, 1.0, 1000);
    std::printf("  RK4 (n = 1000): y(1) = %.10f  y'(1) = %.10f, ODE residual at x = 0.5: %.2e\n", ref.Y[1000][0],
                ref.Y[1000][1], ex11_ode_residual(ref, 500));
    for (std::size_t n : {10u, 100u, 1000u})
        std::printf("    forward Euler n = %-5zu y(1) = %.8f\n", n, ex11_euler({1.0, 0.0}, 0.0, 1.0, n).Y[n][0]);

    header("Ex 12: 1 + sigma^2 S^2/2 * Gamma/Theta (S = 50, K = 52, tau = 0.75, sigma = 30%, q = 0)");
    for (double r : {0.0, 0.01, 0.05, 0.1}) {
        const Ex12Market m12{50.0, 52.0, 0.75, 0.3, r};
        std::printf("    r = %-5.2f lhs = %.12f", r, ex12_lhs(m12));
        if (r != 0.0) std::printf("  1/(1 + sigma N'(d2)/(2 r sqrt(tau) N(d2))) = %.12f", ex12_rhs(m12));
        std::printf("\n");
    }

    header("Ex 13-14: PDE residuals (K = 40, T = 1, sigma = 30%, r = 5%, q = 2%)");
    for (double S : {20.0, 40.0, 60.0})
        std::printf("    S = %-4.0f t = 0.5: call Greeks %.1e, call FD %.1e, forward exact %.1e, forward FD %.1e\n",
                    S, ex13_residual_greeks(S, 0.5), ex13_residual_fd(S, 0.5), ex14_residual_exact(S, 0.5),
                    ex14_residual_fd(S, 0.5));
    return 0;
}
