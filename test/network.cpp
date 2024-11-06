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

    Matrix input(3, 1, 1.0);

    Matrix output = network.predict(input);

    EXPECT_EQ(output.rows(), 4);
    EXPECT_EQ(output.cols(), 1);
}

TEST(Network, Loss)
{
    Network network(4);

    Matrix predicted(4, 1, 1.0);

    predicted[0][0] = 0.5;
    predicted[1][0] = 0.5;
    predicted[2][0] = 0.5;
    predicted[3][0] = 0.5;

    Matrix target(4, 1, 0);

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

void testXor(Network& network)
{
    Matrix input = Matrix::random(network.inputs(), 1);

    Matrix target(network.outputs(), 1, 0);
    target[1][0] = 1;
    target[2][0] = 1;

    network.learn(input, target, 0.5, 150);

    Matrix output = network.predict(input);

    for (int i = 0; i < output.rows(); i++)
    {
        EXPECT_NEAR(output[i][0], target[i][0], 0.1);
    }
}

TEST(Network, SigmoidNetwork)
{
    Network network(3);

    network.add_layer(4, Activation::Sigmoid);
    network.add_layer(4, Activation::Sigmoid);

    testXor(network);
}

TEST(Network, ReLUNetwork)
{
    Network network(3);

    network.add_layer(4, Activation::ReLU);

    testXor(network);
}

TEST(Network, ReLUDeepNetwork)
{
    Network network(3);

    network.add_layer(4, Activation::ReLU);
    network.add_layer(4, Activation::ReLU);

    testXor(network);
}

TEST(Network, TanhNetwork)
{
    Network network(3);

    network.add_layer(4, Activation::Tanh);

    testXor(network);
}
