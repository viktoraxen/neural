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
    : m_weights(Matrix::random(width, inputs))
    , m_biases(Matrix::random(width, 1))
    , m_input(Matrix::random(width, 1))
    , m_activated_output(Matrix::random(width, 1))
    , m_delta(Matrix::random(width, 1))
    , m_inputs(inputs)
    , m_activation(activation)
{}

Matrix Layer::forward(const Matrix& input)
{
    m_input = input;
    Matrix net_input = m_weights.multiply(m_input) + m_biases;

    switch (m_activation)
    {
        case Activation::Sigmoid:
            m_activated_output = net_input.sigmoid();
            break;
        case Activation::Softmax:
            m_activated_output = net_input.softmax();
            break;
        case Activation::ReLU:
            m_activated_output = net_input.relu();
            break;
        case Activation::Tanh:
            m_activated_output = net_input.tanh();
            break;
        default: 
            m_activated_output = net_input;
    }

    return m_activated_output;
}

Matrix Layer::backward(const Matrix& delta)
{
    Matrix activation_derivative;

    switch(m_activation)
    {
        case Activation::Sigmoid: 
            activation_derivative = m_activated_output.sigmoid_derivative();
            break;
        case Activation::Softmax: 
            activation_derivative = m_activated_output;
            break;
        case Activation::ReLU: 
            activation_derivative = m_activated_output.relu_derivative();
            break;
        case Activation::Tanh: 
            activation_derivative = m_activated_output.tanh_derivative();
            break;
        default: 
            activation_derivative = m_activated_output;
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
}

void Layer::print() const
{
    std::cout << m_activation;
    std::cout << "(" << inputs() << " -> " << width() << ")" << std::endl;
}
