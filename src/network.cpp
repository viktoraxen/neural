#include "network.hpp"

#include <stdexcept>
#include <iostream>

Network::Network(int inputs, int depth, int width, int outputs)
    : m_inputs(inputs)
    , m_depth(depth)
    , m_width(width)
    , m_outputs(outputs)
{
    m_layers = std::vector<Layer>();

    m_layers.push_back(Layer(m_inputs, m_width));

    for (int i = 0; i < m_depth - 1; i++)
    {
        m_layers.push_back(Layer(m_width, m_width));
    }

    m_layers.push_back(Layer(m_width, m_outputs));
}

Matrix Network::forward_layer(const Layer& layer, const Matrix& input) const
{
    Matrix stacked_biases = Matrix::stack(layer.biases(), input.rows());

    Matrix a = input.multiply(layer.weights()) + stacked_biases;

    return a.sigmoid();
}

Matrix Network::predict(const Matrix& input, Matrix& output) const
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

void Network::print() const
{
    std::cout << "Layers:" << std::endl;

    for (const auto& layer : m_layers)
    {
        layer.print();
    }
}
