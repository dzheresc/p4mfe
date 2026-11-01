// Sections 3.5-3.6: Black-Scholes prices and Greeks of a call and a put, with
// the closed-form Greeks checked against finite differences and the
// Black-Scholes PDE (section 3.6.1).

#include <p4mfe/ch01/calculus.hpp>
#include <p4mfe/ch03/black_scholes.hpp>

#include <cstdio>

int main() {
    using namespace p4mfe::ch03;
    using p4mfe::ch01::derivative;
    const double S = 40, K = 40, T = 0.5, sigma = 0.3, r = 0.05, q = 0.02;
    std::printf("S = %g, K = %g, T = %g, sigma = %g, r = %g, q = %g\n\n", S, K, T, sigma, r, q);

    std::printf("%-5s %10s %10s %10s %10s %10s %10s %12s\n", "", "price", "delta", "gamma", "vega", "theta",
                "rho", "PDE resid");
    for (OptionType type : {OptionType::Call, OptionType::Put}) {
        const auto g = bs_greeks(type, S, K, T, sigma, r, q);
        std::printf("%-5s %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %12.2e\n",
                    type == OptionType::Call ? "call" : "put", bs_price(type, S, K, T, sigma, r, q), g.delta,
                    g.gamma, g.vega, g.theta, g.rho, bs_pde_residual(type, S, K, T, sigma, r, q));
    }

    const auto g = bs_greeks(OptionType::Call, S, K, T, sigma, r, q);
    const double fd_delta = derivative([&](double s) { return bs_call(s, K, T, sigma, r, q); }, S);
    const double fd_vega = derivative([&](double v) { return bs_call(S, K, T, v, r, q); }, sigma);
    std::printf("\ncall delta: closed form %.10f, finite difference %.10f\n", g.delta, fd_delta);
    std::printf("call vega:  closed form %.10f, finite difference %.10f\n", g.vega, fd_vega);
    std::printf("S e^(-qT) phi(d1) - K e^(-rT) phi(d2) = %.2e\n", bs_phi_identity_gap(S, K, T, sigma, r, q));
}
