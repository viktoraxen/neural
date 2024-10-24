#include <matrix.hpp>

#include <iostream>

enum class Activation
{
    Sigmoid,
    Softmax
};

std::ostream& operator<<(std::ostream& os, Activation color);

using namespace Math;

class Layer
{
public:
    Layer(int inputs, int width, Activation activation);
    ~Layer() = default;

    const Matrix& biases() const { return m_biases; }
    const Matrix& weights() const { return m_weights; }

    Matrix forward(const Matrix& input) const;

    int width() const { return m_weights.cols(); }

    // DEBUG
    void print() const;

private:
    Matrix m_weights;
    Matrix m_biases;
    Activation m_activation;
};
