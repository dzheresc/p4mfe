// Section 7.4: the Black-Scholes PDE checked by finite differences, and its
// financial meaning: a Delta-hedged short option replicates the payoff, with
// an error that shrinks like 1/sqrt(rebalancing frequency) whatever the drift.

#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch07/bs_pde.hpp>

#include <cmath>
#include <cstdio>

int main() {
    using namespace p4mfe;
    using namespace p4mfe::ch07;
    const double S = 100, K = 105, T = 0.5, sigma = 0.2, r = 0.03, q = 0.01;

    std::printf("PDE residual Theta + 1/2 s^2 S^2 Gamma + (r-q) S Delta - r V by finite differences, t = 0\n");
    const auto call = [&](double s, double t) { return ch03::bs_call(s, K, T - t, sigma, r, q); };
    const auto put = [&](double s, double t) { return ch03::bs_put(s, K, T - t, sigma, r, q); };
    const auto fwd = [&](double s, double t) { return forward_contract_value(s, t, K, T, r, q); };
    const auto sq = [&](double s, double) { return s * s; };
    const auto pw = [&](double s, double t) { return power_solution(2.0, s, t, sigma, r, q); };
    std::printf("  call                 %10.2e\n", bs_pde_residual_fd(call, S, 0, sigma, r, q));
    std::printf("  put                  %10.2e\n", bs_pde_residual_fd(put, S, 0, sigma, r, q));
    std::printf("  forward contract     %10.2e\n", bs_pde_residual_fd(fwd, S, 0, sigma, r, q));
    std::printf("  S^2 e^(c t)          %10.2e\n", bs_pde_residual_fd(pw, S, 0, sigma, r, q));
    std::printf("  S^2 (not a solution) %10.2e\n", bs_pde_residual_fd(sq, S, 0, sigma, r, q));

    const double C = ch03::bs_call(S, K, T, sigma, r, q);
    std::printf("\nShort call (C = %.4f) Delta-hedged n times; P&L over 5000 paths\n", C);
    std::printf("%6s  %8s  %12s  %12s\n", "n", "mu", "mean", "std dev");
    for (double mu : {-0.1, 0.03, 0.2})
        for (std::size_t n : {4, 16, 64, 256}) {
            const auto st = delta_hedge_stats(OptionType::Call, S, K, T, sigma, r, q, mu, n, 5000);
            std::printf("%6zu  %8.2f  %12.5f  %12.5f\n", n, mu, st.mean, st.std_dev);
        }
}
