#include "layer.hpp"

std::ostream& operator<<(std::ostream& os, Activation activation)
{
    switch (activation) {
        case Activation::Sigmoid: os << "Sigmoid"; break;
        case Activation::Softmax: os << "Softmax"; break;
        case Activation::ReLU:    os << "ReLU"; break;
        case Activation::Tanh:    os << "Tanh"; break;
    }
    return os;
}

Layer::Layer(int inputs, int width, Activation activation)
    : m_weights(width, inputs, 1.0)
    , m_biases(width, 1, 0.0)
    , m_input(inputs, 1, 0.0)
    , m_net_input(width, 1, 0.0)
    , m_activated_output(width, 1, 0.0)
    , m_delta(width, 1, 0.0)
    , m_activation(activation)
{}

Math::Matrix Layer::forward(const Math::Matrix& input)
{
    m_input = input;
    m_net_input = m_weights.multiply(m_input) + m_biases;

    switch (m_activation)
    {
        case Activation::Sigmoid:
            return m_net_input.sigmoid();
            break;
        case Activation::Softmax:
            return m_net_input.softmax();
            break;
        default: 
            m_activated_output = m_net_input;
    }

    return m_activated_output;
}

Math::Matrix Layer::backward(const Math::Matrix& delta)
{
    Math::Matrix activation_derivative;

    switch(m_activation)
    {
        case Activation::Sigmoid: 
            activation_derivative = m_activated_output.sigmoid_derivative();
            break;
        case Activation::Softmax: 
            activation_derivative = m_activated_output;
            break;
        default: 
            activation_derivative = Math::Matrix::I(m_net_input.rows()); 
    }

    m_delta = delta * activation_derivative;

    return m_weights.T().multiply(m_delta);
}

void Layer::update(double learning_rate)
{
    // delta(l) * a(l-1)^T
    m_weights -= (m_delta * m_input.T()) * learning_rate;
    // delta(l)
    m_biases -= m_delta * learning_rate;

    // std::cout << "new Weights: " << std::endl;
    // m_weights.print();

    // std::cout << "new Biases: " << std::endl;
    // m_biases.print();
}

void Layer::print() const
{
    std::cout << m_activation;
    std::cout << "(" << m_weights.rows() << " -> " << m_weights.cols() << ")" << std::endl;
}
