#include "network.hpp"

#include <stdexcept>
#include <iostream>

#define PRINT_SHAPE(x, name) std::cout << name << " shape: " << x.shape().first << ", " << x.shape().second << std::endl;

Network::Network(int inputs)
    : m_inputs(inputs)
    , m_layers(std::vector<Layer>())
{}

void Network::add_layer(int width, Activation activation)
{
    int inputs = m_layers.empty() ? m_inputs : m_layers.back().width();

    m_layers.push_back(Layer(inputs, width, activation));
}

Matrix Network::predict(const Matrix& input)
{
    if (input.rows() != m_inputs || input.cols() != 1)
        throw std::runtime_error("Invalid input size");

    Matrix current_input = input;

    for (auto& layer : m_layers)
    {
        current_input = layer.forward(current_input);
    }
 
    return current_input;
}

void Network::learn(const Math::Matrix& input, 
                    const Math::Matrix& target, 
                    const double learning_rate,
                    const double epochs)
{
    if (input.rows() != m_inputs)
        throw std::runtime_error("Invalid input size (input size does not match network input size).");

    if (target.rows() != outputs())
        throw std::runtime_error("Invalid output size (target size does not match network output size).");

    for (int i = 0; i < epochs; i++)
    {
        // Predicted output with the current weights and biases
        Matrix output = predict(input);

        // The loss function between the output and the target
        double error = loss(output, target, LossFunction::MeanSquaredError);
        // std::cout << "Epoch " << i << " - Error: " << error << std::endl;

        // Derivative of the loss function with respect to the output
        Matrix delta = output - target;

        for (int i = m_layers.size() - 1; i >= 0; i--)
        {
            Layer& layer = m_layers[i];

            delta = layer.backward(delta);
            layer.update(learning_rate);
        }
    }
}

double Network::loss(const Math::Matrix& output, 
                     const Math::Matrix& target,
                     LossFunction loss_function)
{
    switch (loss_function)
    {
        case LossFunction::MeanSquaredError:
            return (output - target).square_elements().sum_cols()[0][0] / output.rows();
        case LossFunction::CrossEntropy:
            return target.multiply(output.log_elements())[0][0];
        default:
            throw std::runtime_error("Invalid loss function");
    }
}

void Network::print() const
{
    std::cout << "Layers:" << std::endl;

    for (const auto& layer : m_layers)
    {
        std::cout << "  ";
        layer.print();
    }
}
