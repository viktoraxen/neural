#include "network.hpp"

#include <stdexcept>
#include <iostream>
#include <iomanip>

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

void Network::learn(const Matrix& input, 
                    const Matrix& target, 
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

        if (i % 10 == 0)
            std::cout << "Epoch " << i << " - Error: " << error << std::endl;

        // Derivative of the loss function with respect to the output
        Matrix delta = loss_gradient(output, target, LossFunction::MeanSquaredError);

        for (int i = m_layers.size() - 1; i >= 0; i--)
        {
            Layer& layer = m_layers[i];

            delta = layer.backward(delta);
            layer.update(learning_rate);
        }
    }
}

double Network::loss(const Matrix& output, 
                     const Matrix& target,
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

Matrix Network::loss_gradient(const Matrix& output, 
                              const Matrix& target,
                              LossFunction loss_function)
{
    switch (loss_function)
    {
        case LossFunction::MeanSquaredError:
            return output - target;
        case LossFunction::CrossEntropy:
            return Matrix::I(output.rows());
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

        Matrix weights = layer.weights();
        Matrix biases = layer.biases();

        for (int i = 0; i < weights.rows(); i++)
        {
            for (int j = 0; j < weights.cols(); j++)
            {
                std::cout << "    " << std::fixed << std::setprecision(2) << weights[i][j];
            }

            std::cout << "  :  " << std::fixed << std::setprecision(2) << biases[i][0] << std::endl;
        }
    }
}
