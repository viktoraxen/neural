#include <matrix.hpp>

#include <iostream>

enum class Activation
{
    Sigmoid,
    Softmax,
    ReLU,
    Tanh
};

std::ostream& operator<<(std::ostream& os, Activation activation);

using namespace Math;

class Layer
{
public:
    Layer(int inputs, int width, Activation activation);
    ~Layer() = default;

    const Matrix& biases() const { return m_biases; }
    const Matrix& weights() const { return m_weights; }
    const Matrix& activated_output() const { return m_activated_output; }
    const Matrix& z() const { return activated_output(); }

    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& delta);
    void update(double learning_rate);

    int width() const { return m_weights.rows(); }
    int inputs() const { return m_inputs; }

    // DEBUG
    void print() const;

private:
    int m_inputs;
    Matrix m_weights;
    Matrix m_biases;
    Matrix m_input;
    Matrix m_activated_output;
    Matrix m_delta;
    Activation m_activation;
};
