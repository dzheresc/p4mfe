// Sections 2.6-2.10: a 3-year semiannual 6% coupon bond priced at 101.
// Newton's method gives its yield; then duration, convexity, and how well
// they predict the price after a yield move. Numbers are illustrative.

#include <cstdio>
#include <p4mfe/ch02/bonds.hpp>

int main() {
  using namespace p4mfe::ch02;
  const auto cf = bond_cash_flows(3.0, 0.06, 2);
  const double price = 101.0;

  std::printf("Cash flows:\n");
  for (std::size_t i = 0; i < cf.times.size(); ++i)
    std::printf("  t = %.2f  c = %.2f\n", cf.times[i], cf.amounts[i]);

  const double y = bond_yield(cf, price);
  const auto m = bond_metrics(cf, y);
  std::printf("\nYield (continuous)  = %.8f\n", y);
  std::printf("Yield (semiannual)  = %.8f\n", discrete_from_continuous(y, 2));
  std::printf("Duration            = %.6f\n", m.duration);
  std::printf("Convexity           = %.6f\n", m.convexity);
  std::printf("DV01                = %.6f\n", dv01(cf, y));

  std::printf("\n%8s  %12s  %12s  %12s\n", "dy", "exact", "duration",
              "dur+conv");
  for (double dy : {-0.02, -0.005, 0.001, 0.005, 0.02}) {
    std::printf(
        "%8.3f  %12.6f  %12.6f  %12.6f\n", dy, bond_price(cf, y + dy),
        price_approx_duration(m.price, m.duration, dy),
        price_approx_duration_convexity(m.price, m.duration, m.convexity, dy));
  }
}
