#include "math.hpp"

#include <numeric>

double Math::sigmoid(double a)
{ 
    return 1 / (1 + exp(-a)); 
}

void Math::softmax(double* A, double* Z, int size)
{
    double E[size];
    double E_sum = 0;

    for (int i = 0; i < size; i++)
    {
        E[i] = exp(A[i]);
        E_sum += E[i];
    }

    for (int i = 0; i < size; i++)
    {
        Z[i] = E[i] / E_sum;
    }
}

double* Math::softmax(double* A, int size)
{
    double* Z = new double[size];

    softmax(A, Z, size);

    return Z;
}

double Math::cross_entropy(double* A, double* B, int size)
{
    double sum = 0;

    for (int i = 0; i < size; i++)
    {
        sum += A[i] * log(B[i]);
    }

    return -sum / size;
}
