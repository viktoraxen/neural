#include "math.hpp"

#include <cstdlib>
#include <ctime>

double Math::sigmoid(double a)
{ 
    return 1 / (1 + exp(-a)); 
}

double Math::sigmoid_derivative(double a)
{ 
    double s = sigmoid(a);
    return s * (1 - s); 
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

double Math::relu(double a)
{ 
    return a > 0 ? a : 0; 
}

double Math::relu_derivative(double a)
{ 
    return a > 0 ? 1 : 0; 
}

double Math::tanh(double a)
{ 
    return std::tanh(a); 
}

double Math::tanh_derivative(double a)
{ 
    double t = tanh(a);
    return 1 - t * t; 
}

double Math::cross_entropy(double* A, double* B, int size)
{
    double sum = 0;

    for (int i = 0; i < size; i++)
    {
        sum += A[i] * log(B[i]);
    }

    return -sum;
}

double Math::random(double min, double max)
{
    static bool seeded = false;

    if (!seeded)
    {
        srand(time(NULL));
        seeded = true;
    }

    return min + (max - min) * rand() / RAND_MAX;
}
