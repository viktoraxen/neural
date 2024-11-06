#ifndef MATH_HPP
#define MATH_HPP

#include <cmath>

namespace Math
{
    double sigmoid(double a); 
    double sigmoid_derivative(double a);

    void    softmax(double* A, double* Z, int size);
    double* softmax(double* A, int size);

    double relu(double a);
    double relu_derivative(double a);

    double tanh(double a);
    double tanh_derivative(double a);

    double cross_entropy(double* A, double* B, int size);

    double random(double min = 0.0, double max = 1.0);
}

#endif // MATH_HPP
