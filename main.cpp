#include <iostream>
#include "network.hpp"

int main(int argc, char **argv) 
{
    Network network(4, 10, 3, 4);
    network.print();

    Math::Matrix input = Math::Matrix::filled(1, 4, 1.0);
    Math::Matrix output;

    network.predict(input, output);

    std::cout << std::endl;
    output.print();

    return 0;
}
