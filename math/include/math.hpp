#ifndef MATH_HPP
#define MATH_HPP

#include <cmath>
#include <vector>

namespace Math
{
    double sigmoid(double a); 
    std::vector<double> softmax(std::vector<double> A);
    std::vector<double> cross_entropy(std::vector<double> A, std::vector<double> B);
}

#endif // MATH_HPP
