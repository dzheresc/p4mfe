// Sections 5.3-5.5 in one place: bond yields, implied volatility, and a zero
// curve bootstrapped from a sparse set of bonds with linear interpolation.
// Numbers are illustrative.

#include <cstdio>
#include <p4mfe/ch02/bonds.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch05/bond_yield.hpp>
#include <p4mfe/ch05/bootstrapping.hpp>
#include <p4mfe/ch05/implied_vol.hpp>

int main() {
  using namespace p4mfe;
  using namespace p4mfe::ch05;

  // 5.3: 3-year semiannual 6% bond at 101.
  const auto cf = ch02::bond_cash_flows(3.0, 0.06, 2);
  const auto yn = bond_yield_newton(cf, 101.0);
  const auto ys = bond_yield_secant(cf, 101.0);
  const auto yb = bond_yield_bisection(cf, 101.0);
  std::printf(
      "Bond yield:  Newton %.12f (%zu it)  secant %.12f (%zu it)  bisection "
      "%.12f (%zu it)\n",
      yn.x, yn.iterations, ys.x, ys.iterations, yb.x, yb.iterations);

  // 5.4: a call worth 2.5 with S = 30, K = 30, T = 0.5, r = 3%, q = 1%.
  const auto vn =
      implied_vol_newton(OptionType::Call, 2.5, 30, 30, 0.5, 0.03, 0.01);
  const auto vs =
      implied_vol_secant(OptionType::Call, 2.5, 30, 30, 0.5, 0.03, 0.01);
  std::printf("Implied vol: Newton %.12f (%zu it)  secant %.12f (%zu it)\n",
              vn.x, vn.iterations, vs.x, vs.iterations);

  // 5.5: overnight rate 5%; semiannual bonds maturing at 0.5, 1, 3 and 5 years.
  const std::vector<BondQuote> quotes{{0.5, 0.0, 97.5},
                                      {1.0, 0.05, 100.0},
                                      {3.0, 0.055, 100.5},
                                      {5.0, 0.06, 101.0}};
  const auto curve = bootstrap_zero_curve(quotes, 2, ZeroCurve{{0.0}, {0.05}});
  std::printf("\nBootstrapped zero curve (linear between nodes):\n%8s  %10s\n",
              "t", "r(0,t)");
  for (std::size_t j = 0; j < curve.times.size(); ++j)
    std::printf("%8.2f  %10.6f\n", curve.times[j], curve.rates[j]);
  std::printf("\nRepricing check:\n");
  for (const auto& q : quotes) {
    const auto c = ch02::bond_cash_flows(q.maturity, q.coupon_rate, 2);
    std::printf("  T = %.1f  quote %.4f  model %.10f\n", q.maturity, q.price,
                ch02::bond_price_from_zero_rate(
                    c, [&](double t) { return zero_rate_linear(curve, t); }));
  }
}
