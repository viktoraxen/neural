#include "math.hpp"

#include <numeric>

double Math::sigmoid(double a)
{ 
    return 1 / (1 + exp(-a)); 
}

std::vector<double> Math::softmax(std::vector<double> A)
{
    std::vector<double> E(A.size());

    for (int i = 0; i < A.size(); i++)
    {
        E[i] = exp(A[i]);
    }

    double E_sum = std::reduce(E.begin(), E.end(), 0.0);

    std::vector<double> Z(E.size());

    for (int i = 0; i < A.size(); i++)
    {
        Z[i] = E[i] / E_sum;
    }

    return Z;
}

// std::vector<double> Math::cross_entropy(std::vector<double> A, std::vector<double> B)
// {
//     // if (A.size() != B.size())
//     //     throw std::invalid_argument("A and B must have the same size");
//
//     
// }
