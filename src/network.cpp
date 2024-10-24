#include "network.hpp"

#include <stdexcept>
#include <iostream>

Network::Network(int inputs)
    : m_inputs(inputs)
    , m_layers(std::vector<Layer>())
{}

void Network::add_layer(int width, Activation activation)
{
    int inputs = m_layers.empty() ? m_inputs : m_layers.back().width();

    m_layers.push_back(Layer(inputs, width, activation));
}

Matrix Network::predict(const Matrix& input) const
{
    if (input.cols() != m_inputs || input.rows() != 1)
        throw std::runtime_error("Invalid input size");

    Matrix current_input = input;

    for (const auto& layer : m_layers)
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
    if (input.cols() != m_inputs)
        throw std::runtime_error("Invalid input size (input size does not match network input size).");

    if (target.cols() != outputs())
        throw std::runtime_error("Invalid output size (target size does not match network output size).");

    if (input.rows() != target.rows())
        throw std::runtime_error("Invalid batch size (input size does not match target size).");

    for (int i = 0; i < epochs; i++)
    {
        Matrix output = predict(input);
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
