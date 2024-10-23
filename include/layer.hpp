#include <matrix.hpp>

using namespace Math;

class Layer
{
public:
    Layer(int inputs, int width);
    ~Layer() = default;

    const Matrix& biases() const { return m_biases; }
    const Matrix& weights() const { return m_weights; }

    Matrix forward(const Matrix& input) const;

    // DEBUG
    void print() const;

private:
    Matrix m_weights;
    Matrix m_biases;
};
