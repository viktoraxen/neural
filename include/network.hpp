#include <layer.hpp>

using namespace Math;

enum class LossFunction
{
    MeanSquaredError,
    CrossEntropy
};

class Network
{
public:
    Network(int inputs);
    ~Network() = default;

    void add_layer(
        int        width,
        Activation activation);

    Matrix predict(const Math::Matrix& input);
    void learn(const Math::Matrix& input,
               const Math::Matrix& target,
               const double        learning_rate,
               const double        epochs);

    const Layer& layer(int index) const { return m_layers[index]; }

    int depth() const { return m_layers.size(); }
    int inputs() const { return m_inputs; }
    int outputs() const { return m_layers.empty() ? 0 : m_layers.back().width(); }

    // DEBUG
    void print() const;

    double loss(const Math::Matrix& output, 
                const Math::Matrix& target,
                LossFunction loss_function);
private:
    int m_inputs;
    std::vector<Layer> m_layers;

};
