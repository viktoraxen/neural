#ifndef MATH_HPP
#define MATH_HPP

#include <cmath>

namespace Math
{
    double sigmoid(double a); 

    void    softmax(double* A, double* Z, int size);
    double* softmax(double* A, int size);

    double cross_entropy(double* A, double* B, int size);
}

#endif // MATH_HPP
