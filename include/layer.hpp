#include <math.hpp>

class Layer
{
public:
    Layer(int inputs, int width);
    ~Layer() = default;

    Math::Matrix forward(const Math::Matrix& input) const;

    // DEBUG
    void print() const;

private:
    Math::Matrix m_weights;
    Math::Matrix m_biases;
};
