// Sections 8.9 and 8.10: barrier option prices by the method of images, and
// the (non-)value of early exercise.

#include <cstdio>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch08/barrier.hpp>
#include <p4mfe/ch08/early_exercise.hpp>

int main() {
  using namespace p4mfe;
  using namespace p4mfe::ch08;
  const double K = 100, T = 1, sigma = 0.3, r = 0.05, q = 0.0;

  std::printf(
      "Down-and-out / down-and-in calls, K = %.0f, T = %.0f, sigma = %.2f, r = "
      "%.2f\n",
      K, T, sigma, r);
  std::printf("%6s  %6s  %10s  %10s  %10s\n", "B", "S", "vanilla", "out", "in");
  for (double B : {70.0, 90.0, 110.0})
    for (double S : {B + 5, 120.0})
      std::printf("%6.0f  %6.0f  %10.5f  %10.5f  %10.5f\n", B, S,
                  ch03::BsCall(S, K, T, sigma, r, q),
                  down_and_out_call(S, K, B, T, sigma, r, q),
                  down_and_in_call(S, K, B, T, sigma, r, q));

  std::printf("\nUp-and-out puts, S = 100\n%6s  %10s  %10s\n", "B", "vanilla",
              "out");
  for (double B : {105.0, 120.0, 150.0})
    std::printf("%6.0f  %10.5f  %10.5f\n", B,
                ch03::BsPut(100, K, T, sigma, r, q),
                up_and_out_put(100, K, B, T, sigma, r, q));

  std::printf(
      "\nEarly exercise premium (American - European, 1000-step tree), S = K = "
      "100\n");
  std::printf("%6s  %6s  %12s  %12s\n", "q", "type", "European", "premium");
  for (double qq : {0.0, 0.04, 0.1})
    for (OptionType type : {OptionType::kCall, OptionType::kPut})
      std::printf("%6.2f  %6s  %12.6f  %12.6f\n", qq,
                  type == OptionType::kCall ? "call" : "put",
                  binomial_price(type, 100, K, T, sigma, r, qq, 1000,
                                 ExerciseStyle::European),
                  early_exercise_premium(type, 100, K, T, sigma, r, qq, 1000));

  std::printf(
      "\nExercising a call now (q = 0) loses C - (S - K) >= K (1 - e^(-r tau)) "
      "= %.4f for tau = 1\n",
      call_exercise_loss_bound(K, 1.0, r));
  for (double S : {100.0, 150.0, 300.0})
    std::printf("  S = %5.0f: loss %.4f\n", S,
                call_exercise_loss(S, K, 1.0, sigma, r));
}
