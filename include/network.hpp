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

    Matrix predict(const Matrix& input);
    void learn(const Matrix& input,
               const Matrix& target,
               const double        learning_rate,
               const double        epochs);

    const Layer& layer(int index) const { return m_layers[index]; }

    int depth() const { return m_layers.size(); }
    int inputs() const { return m_inputs; }
    int outputs() const { return m_layers.empty() ? 0 : m_layers.back().width(); }

    // DEBUG
    void print() const;

    double loss(const Matrix& output, 
                const Matrix& target,
                LossFunction loss_function);

    Matrix loss_gradient(const Matrix& output, 
                         const Matrix& target,
                         LossFunction loss_function);
private:
    int m_inputs;
    std::vector<Layer> m_layers;

};
