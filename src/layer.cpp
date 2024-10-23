#include "layer.hpp"

#include <iostream>

Layer::Layer(int inputs, int width)
    : m_weights(Math::Matrix::filled(inputs, width, 1.0))
    , m_biases(Math::Matrix::filled(1, width, 0.0))
{}

Math::Matrix Layer::forward(const Math::Matrix& input) const
{
    Math::Matrix stacked_biases = Math::Matrix::stack(m_biases, input.rows());

    Math::Matrix a = input.multiply(m_weights) + stacked_biases;

    return a.sigmoid();
}

void Layer::print() const
{
    std::cout << "Weights:" << std::endl;
    m_weights.print();
    std::cout << "Biases:" << std::endl;
    m_biases.print();
}
