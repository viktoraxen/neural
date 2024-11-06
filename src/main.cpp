#include "network.hpp"

int main(int argc, char **argv) {
    int input_size = 3;
    int output_size = 4;
    Network network(input_size);

    network.add_layer(4, Activation::ReLU);
    network.add_layer(output_size, Activation::ReLU);

    network.print();

    Matrix input(input_size, 1, 1.0);

    Matrix output = network.predict(input);

    std::cout << "Output before training:" << std::endl;
    output.print();

    Matrix target(output_size, 1, 0);
    target[1][0] = 1;
    target[2][0] = 1;
    target[3][0] = 1;

    network.learn(input, target, 0.5, 100);

    output = network.predict(input);

    std::cout << "Output after training:" << std::endl;
    output.print();

    network.print();

    return 0;
}
