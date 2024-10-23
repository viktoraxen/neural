#include "layer.hpp"

class Network
{
public:
    Network(int inputs, int depth, int width, int outputs);
    ~Network() = default;

    Math::Matrix predict(const Math::Matrix& input, Math::Matrix& output) const;

    const Layer& layer(int index) const { return m_layers[index]; }

    // DEBUG
    void print() const;

private:
    int m_inputs;
    int m_outputs;
    int m_depth;
    int m_width;
    std::vector<Layer> m_layers;
};
