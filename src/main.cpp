#include "network.hpp"

#include <iostream>

int main(int argc, char **argv) {
    int input_size = 3;
    Network network(input_size);

    network.add_layer(4, Activation::Sigmoid);
    network.add_layer(5, Activation::Sigmoid);
    network.add_layer(2, Activation::Softmax);
    network.print();

    Math::Matrix input(1, input_size, 1.0);
    Math::Matrix output = network.predict(input);

    std::cout << std::endl;
    output.print();

    return 0;
}
