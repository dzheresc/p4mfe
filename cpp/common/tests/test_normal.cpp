#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <p4mfe/normal.hpp>

namespace {

TEST(Normal, PdfAtZero) {
  EXPECT_NEAR(p4mfe::NormPdf(0.0), 0.3989422804014327, 1e-15);
}

TEST(Normal, CdfKnownValues) {
  EXPECT_DOUBLE_EQ(p4mfe::NormCdf(0.0), 0.5);
  EXPECT_NEAR(p4mfe::NormCdf(1.959963984540054), 0.975, 1e-12);
  EXPECT_NEAR(p4mfe::NormCdf(-1.0), 0.15865525393145707, 1e-15);
}

TEST(Normal, CdfSymmetry) {
  for (double x : {0.1, 0.5, 1.0, 2.5, 4.0}) {
    EXPECT_NEAR(p4mfe::NormCdf(x) + p4mfe::NormCdf(-x), 1.0, 1e-15) << x;
  }
}

TEST(Normal, InverseCdf) {
  EXPECT_DOUBLE_EQ(p4mfe::NormInv(0.5), 0.0);
  EXPECT_NEAR(p4mfe::NormInv(0.975), 1.959963984540054, 1e-14);
  EXPECT_NEAR(p4mfe::NormInv(0.995), 2.5758293035489004, 1e-14);
  for (double p :
       {1e-12, 1e-6, 0.01, 0.02425, 0.2, 0.5, 0.7, 0.97575, 0.999, 1 - 1e-9}) {
    const double x = p4mfe::NormInv(p);
    EXPECT_NEAR(p4mfe::NormCdf(x), p, 1e-13 * std::min(p, 1 - p) + 2.3e-16)
        << p;
    // 1 - p is exact enough only when p is not tiny.
    if (p >= 1e-6) {
      EXPECT_NEAR(p4mfe::NormInv(1 - p), -x, 1e-9) << p;
    }
  }
  EXPECT_EQ(p4mfe::NormInv(0.0), -std::numeric_limits<double>::infinity());
  EXPECT_EQ(p4mfe::NormInv(1.0), std::numeric_limits<double>::infinity());
}

}  // namespace
