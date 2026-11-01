// Workload W1 (docs/performance-testing-plan.md): Black-Scholes pricing,
// Greeks and implied volatility over an index option chain.
//
// Every benchmark sweeps the whole chain (or one slice of it) per iteration
// and reports options/second. Implied vol benchmarks also report the mean
// iteration count. Setup checks once that implied vol recovers the vol each
// price was generated with, so a fast wrong answer fails loudly.
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch03/implied_vol.hpp>

#include <benchmark/benchmark.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace {

using p4mfe::bench::OptionQuote;
using p4mfe::ch01::OptionType;

// Slices of the chain that take different paths through the solvers.
enum class Slice { All, Atm, Wings, Short };

const char* slice_name(Slice s) {
    switch (s) {
        case Slice::All: return "all";
        case Slice::Atm: return "atm";
        case Slice::Wings: return "wings";
        case Slice::Short: return "short";
    }
    return "?";
}

bool in_slice(const OptionQuote& o, Slice s) {
    switch (s) {
        case Slice::All: return true;
        case Slice::Atm: return std::abs(o.log_moneyness) < 0.05;
        case Slice::Wings: return std::abs(o.log_moneyness) > 0.3;
        case Slice::Short: return o.T <= 1.0 / 12.0;
    }
    return false;
}

// All data is built before any benchmark runs (forced from the registration
// below), so the allocation counters measure the kernels, not the setup.
struct CallPut {
    const OptionQuote* call;
    double call_price, put_price;
};

struct Market {
    p4mfe::bench::ChainSpec spec;
    std::vector<OptionQuote> chain;
    std::vector<OptionQuote> slices[4];
    std::vector<CallPut> pairs;  // call/put with the same underlying, strike and expiry
};

const Market& market() {
    static const Market m = [] {
        Market x;
        x.spec = p4mfe::bench::chain_spec_for_tier();
        x.chain = p4mfe::bench::make_option_chain(x.spec);
        for (const Slice s : {Slice::All, Slice::Atm, Slice::Wings, Slice::Short})
            for (const auto& o : x.chain)
                if (in_slice(o, s)) x.slices[static_cast<int>(s)].push_back(o);
        for (std::size_t i = 0; i + 1 < x.chain.size(); ++i) {
            const auto& c = x.chain[i];
            const auto& p = x.chain[i + 1];
            if (c.type == OptionType::Call && p.type == OptionType::Put && c.K == p.K && c.T == p.T && c.S == p.S)
                x.pairs.push_back({&c, c.price, p.price});
        }
        return x;
    }();
    return m;
}

void set_items_options(benchmark::State& state, std::size_t per_iteration) {
    state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                            static_cast<std::int64_t>(per_iteration));
    state.counters["options"] = static_cast<double>(per_iteration);
}

// ------------------------------------------------------------ pricing ------

void BM_bs_price(benchmark::State& state) {
    const auto& m = market();
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& o : m.chain) sum += p4mfe::ch03::bs_price(o.type, o.S, o.K, o.T, o.sigma, m.spec.r, m.spec.q);
        benchmark::DoNotOptimize(sum);
    }
    set_items_options(state, m.chain.size());
}

void BM_bs_greeks(benchmark::State& state) {
    const auto& m = market();
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& o : m.chain) {
            const auto g = p4mfe::ch03::bs_greeks(o.type, o.S, o.K, o.T, o.sigma, m.spec.r, m.spec.q);
            sum += g.delta + g.gamma + g.vega + g.theta + g.rho;
        }
        benchmark::DoNotOptimize(sum);
    }
    set_items_options(state, m.chain.size());
}

// Put-call parity residual (ch01) for every call/put pair in the chain.
void BM_parity_violation(benchmark::State& state) {
    const auto& m = market();
    for (const auto& cp : m.pairs) {
        const auto& o = *cp.call;
        const double v = p4mfe::ch01::parity_violation(cp.call_price, cp.put_price, o.S, o.K, o.T, m.spec.r, m.spec.q);
        if (std::abs(v) > 1e-8 * o.S) {
            state.SkipWithError("put-call parity does not hold on model prices");
            return;
        }
    }
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& cp : m.pairs) {
            const auto& o = *cp.call;
            sum += p4mfe::ch01::parity_violation(cp.call_price, cp.put_price, o.S, o.K, o.T, m.spec.r, m.spec.q);
        }
        benchmark::DoNotOptimize(sum);
    }
    set_items_options(state, m.pairs.size());
}

// --------------------------------------------------------- implied vol -----

using IvSolver = p4mfe::ch03::ImpliedVolResult (*)(OptionType, double, double, double, double, double, double);

p4mfe::ch03::ImpliedVolResult iv_newton(OptionType t, double p, double S, double K, double T, double r, double q) {
    return p4mfe::ch03::implied_vol_newton(t, p, S, K, T, r, q);
}
p4mfe::ch03::ImpliedVolResult iv_bisection(OptionType t, double p, double S, double K, double T, double r,
                                           double q) {
    return p4mfe::ch03::implied_vol_bisection(t, p, S, K, T, r, q);
}

void BM_implied_vol(benchmark::State& state, IvSolver solve, Slice s) {
    const auto& m = market();
    const auto& quotes = m.slices[static_cast<int>(s)];
    if (quotes.empty()) {
        state.SkipWithError("empty slice");
        return;
    }
    // Correctness check, once, outside the timed loop.
    std::size_t iterations = 0;
    for (const auto& o : quotes) {
        try {
            const auto res = solve(o.type, o.price, o.S, o.K, o.T, m.spec.r, m.spec.q);
            // Recovered price within a hundredth of a tick, or vol within 1e-6.
            const double repriced = p4mfe::ch03::bs_price(o.type, o.S, o.K, o.T, res.sigma, m.spec.r, m.spec.q);
            if (std::abs(res.sigma - o.sigma) > 1e-6 && std::abs(repriced - o.price) > 1e-2 * m.spec.tick) {
                state.SkipWithError("implied vol does not recover the generating vol");
                return;
            }
            iterations += res.iterations;
        } catch (const std::exception& e) {
            state.SkipWithError(std::string("implied vol threw: ") + e.what());
            return;
        }
    }
    for (auto _ : state) {
        double sum = 0.0;
        for (const auto& o : quotes) sum += solve(o.type, o.price, o.S, o.K, o.T, m.spec.r, m.spec.q).sigma;
        benchmark::DoNotOptimize(sum);
    }
    set_items_options(state, quotes.size());
    state.counters["mean_iterations"] = static_cast<double>(iterations) / static_cast<double>(quotes.size());
}

const bool registered = [] {
    (void)market();
    benchmark::RegisterBenchmark("W1/ch03/bs_price", BM_bs_price);
    benchmark::RegisterBenchmark("W1/ch03/bs_greeks", BM_bs_greeks);
    benchmark::RegisterBenchmark("W1/ch01/parity_violation", BM_parity_violation);
    for (const Slice s : {Slice::All, Slice::Atm, Slice::Wings, Slice::Short}) {
        benchmark::RegisterBenchmark(std::string("W1/ch03/implied_vol_newton/") + slice_name(s), BM_implied_vol,
                                     &iv_newton, s);
        benchmark::RegisterBenchmark(std::string("W1/ch03/implied_vol_bisection/") + slice_name(s), BM_implied_vol,
                                     &iv_bisection, s);
    }
    return true;
}();

}  // namespace
