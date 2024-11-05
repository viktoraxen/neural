#include <math.hpp>
#include <gtest/gtest.h>

using namespace Math;

TEST(Math, Sigmoid)
{
    EXPECT_EQ(0.5, sigmoid(0));
    EXPECT_EQ(0.7310585786300049, sigmoid(1));
    EXPECT_EQ(0.2689414213699951, sigmoid(-1));
}

TEST(Math, Softmax)
{
    std::vector<double> input = {1, 2, 3};
    std::vector<double> output(input.size());
    softmax(input.data(), output.data(), input.size());

    EXPECT_EQ(0.09003057317038046, output[0]);
    EXPECT_EQ(0.24472847105479767, output[1]);
    EXPECT_EQ(0.6652409557748219, output[2]);
}

TEST(Math, CrossEntropy)
{
    std::vector<double> A = {0.0, 1.0, 0.0};
    std::vector<double> B = {0.2, 0.7, 0.1};

    EXPECT_EQ(-log(0.7), cross_entropy(A.data(), B.data(), A.size()));
}
