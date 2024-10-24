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
    std::vector<double> output = softmax(input);

    EXPECT_EQ(0.09003057317038046, output[0]);
    EXPECT_EQ(0.24472847105479767, output[1]);
    EXPECT_EQ(0.6652409557748219, output[2]);
}
