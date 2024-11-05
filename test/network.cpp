#include <network.hpp>
#include <gtest/gtest.h>

TEST(Network, Initialization)
{
    Network network(3);

    EXPECT_EQ(network.inputs(), 3);
    EXPECT_EQ(network.outputs(), 0);
    EXPECT_EQ(network.depth(), 0);
}

TEST(Network, AddLayer)
{
    Network network(3);

    network.add_layer(5, Activation::Sigmoid);
    network.add_layer(5, Activation::Sigmoid);
    network.add_layer(4, Activation::Sigmoid);

    EXPECT_EQ(network.depth(), 3);
    EXPECT_EQ(network.outputs(), 4);
}

TEST(Network, Predict)
{
    Network network(3);

    network.add_layer(5, Activation::Sigmoid);
    network.add_layer(5, Activation::Sigmoid);
    network.add_layer(4, Activation::Sigmoid);

    Math::Matrix input(3, 1, 1.0);

    Math::Matrix output = network.predict(input);

    EXPECT_EQ(output.rows(), 4);
    EXPECT_EQ(output.cols(), 1);
}

TEST(Network, Loss)
{
    Network network(4);

    Math::Matrix predicted(4, 1, 1.0);

    predicted[0][0] = 0.5;
    predicted[1][0] = 0.5;
    predicted[2][0] = 0.5;
    predicted[3][0] = 0.5;

    Math::Matrix target(4, 1, 0);

    target[0][0] = 1;
    target[3][0] = 1;

    double error = network.loss(predicted, target, LossFunction::MeanSquaredError);

    EXPECT_EQ(error, 0.25);

    predicted[0][0] = 0.5;
    predicted[1][0] = 0.1;
    predicted[2][0] = 0.8;
    predicted[3][0] = 0.35;

    target[0][0] = 1;
    target[1][0] = 0;
    target[2][0] = 1;
    target[3][0] = 0;

    error = network.loss(predicted, target, LossFunction::MeanSquaredError);

    EXPECT_EQ(error, 0.105625);
}
