#include "network.hpp"

int main(int argc, char **argv) {
    int input_size = 3;
    int output_size = 4;
    Network network(input_size);

    network.add_layer(5, Activation::Sigmoid);
    network.add_layer(5, Activation::Sigmoid);
    network.add_layer(output_size, Activation::Sigmoid);

    network.print();

    Math::Matrix input(input_size, 1, 1.0);

    Math::Matrix output = network.predict(input);

    std::cout << "Output before training:" << std::endl;
    output.print();

    Math::Matrix target(output_size, 1, 0);
    target[0][0] = 1;
    target[3][0] = 1;

    network.learn(input, target, 0.5, 100);

    output = network.predict(input);

    std::cout << "Output after training:" << std::endl;
    output.print();

    return 0;
}
