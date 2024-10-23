#ifndef MATH_HPP
#define MATH_HPP

#include <cmath>

namespace Math
{
    static double sigmoid(double a) { return 1 / (1 + exp(-a)); }
}

#endif // MATH_HPP
