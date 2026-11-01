#include <p4mfe/normal.hpp>

#include <gtest/gtest.h>

namespace {

TEST(Normal, PdfAtZero) {
    EXPECT_NEAR(p4mfe::norm_pdf(0.0), 0.3989422804014327, 1e-15);
}

TEST(Normal, CdfKnownValues) {
    EXPECT_DOUBLE_EQ(p4mfe::norm_cdf(0.0), 0.5);
    EXPECT_NEAR(p4mfe::norm_cdf(1.959963984540054), 0.975, 1e-12);
    EXPECT_NEAR(p4mfe::norm_cdf(-1.0), 0.15865525393145707, 1e-15);
}

TEST(Normal, CdfSymmetry) {
    for (double x : {0.1, 0.5, 1.0, 2.5, 4.0}) {
        EXPECT_NEAR(p4mfe::norm_cdf(x) + p4mfe::norm_cdf(-x), 1.0, 1e-15) << x;
    }
}

}  // namespace
