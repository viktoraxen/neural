#include "layer.hpp"

using namespace Math;

class Network
{
public:
    Network(int inputs);
    ~Network() = default;

    void add_layer(
        int        width,
        Activation activation);

    Matrix predict(const Math::Matrix& input) const;
    void learn(const Math::Matrix& input,
               const Math::Matrix& target,
               const double        learning_rate,
               const double        epochs);

    const Layer& layer(int index) const { return m_layers[index]; }

    int depth() const { return m_layers.size() + 1; }
    int outputs() const { return m_layers.back().width(); }

    // DEBUG
    void print() const;

private:
    int m_inputs;
    std::vector<Layer> m_layers;
};
