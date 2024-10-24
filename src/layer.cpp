#include "layer.hpp"

std::ostream& operator<<(std::ostream& os, Activation color)
{
    switch (color) {
        case Activation::Sigmoid: os << "Sigmoid"; break;
        case Activation::Softmax: os << "Softmax"; break;
    }
    return os;
}

Layer::Layer(int inputs, int width, Activation activation)
    : m_weights(inputs, width, 1.0)
    , m_biases(1, width, 0.0)
    , m_activation(activation)
{}

Math::Matrix Layer::forward(const Math::Matrix& input) const
{
    Matrix a = input.multiply(m_weights) + m_biases;

    switch (m_activation)
    {
        case Activation::Sigmoid: return a.sigmoid();
        case Activation::Softmax: return a.softmax();
        default : return a;
    }
}

void Layer::print() const
{
    std::cout << m_activation;
    std::cout << "(" << m_weights.rows() << " -> " << m_weights.cols() << ")" << std::endl;
}
