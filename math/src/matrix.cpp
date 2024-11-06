#include "matrix.hpp"

#include "math.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>

#define UNARY(op) [](double a) { return op(a); }

#define LAMBDA_ADD [](double a, double b) { return a + b; }
#define LAMBDA_SUB [](double a, double b) { return a - b; }
#define LAMBDA_MUL [](double a, double b) { return a * b; }
#define LAMBDA_DIV [](double a, double b) { return a / b; }

namespace Math
{
    Matrix Matrix::identity(int size)
    {
        Matrix result(size, size);

        for (int i = 0; i < size; i++)
        {
            result.elem(i, i) = 1;
        }

        return result;
    }

    Matrix Matrix::cross(const Matrix& a, const Matrix& b)
    {
        if (a.cols() != b.rows())
            throw std::runtime_error("Matrix dimensions do not match");

        return a.T().multiply(b);
    }

    Matrix Matrix::random(int rows, int cols, double min, double max)
    {
        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                result.elem(i, j) = Math::random(min, max);
            }
        }

        return result;
    }

    Matrix::Matrix()
        : m_rows(0)
        , m_cols(0)
    {}

    // Constructor
    Matrix::Matrix(int rows, int cols, double val)
        : m_rows(rows)
        , m_cols(cols)
    {
        m_data = new double[m_rows * m_cols];

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                elem(i, j) = val;
            }
        }
    }

    // Copy constructor
    Matrix::Matrix(const Matrix& other)
    {
        copy(other);
    }

    // Assignment operator
    Matrix& Matrix::operator=(const Matrix& other)
    {
        if (this == &other)
            return *this;

        destroy();
        copy(other);

        return *this;
    }

    // Move constructor
    Matrix::Matrix(Matrix&& other) noexcept
    {
        move(other);
    }

    // Move assignment operator
    Matrix& Matrix::operator=(Matrix&& other) noexcept
    {
        if (this == &other)
            return *this;

        move(other);

        return *this;
    }

    Matrix::~Matrix()
    {
        destroy();
    }

    bool Matrix::operator==(const Matrix& other) const
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            return false;

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                if (elem(i, j) != other.elem(i, j))
                    return false;
            }
        }

        return true;
    }

    int Matrix::determinant() const
    {
        assert(m_rows == m_cols);

        int result = 0;

        if (m_rows == 1)
        {
            result = elem(0, 0);
        }
        else if (m_rows == 2)
        {
            result = elem(0, 0) * elem(1, 1) - elem(0, 1) * elem(1, 0);
        }
        else
        {
            for (int i = 0; i < m_rows; i++)
            {
                Matrix submatrix(m_rows - 1, m_cols - 1);

                for (int j = 1; j < m_rows; j++)
                {
                    for (int k = 0; k < m_cols; k++)
                    {
                        if (k < i)
                        {
                            submatrix.elem(j - 1, k) = elem(j, k);
                        }
                        else if (k > i)
                        {
                            submatrix.elem(j - 1, k - 1) = elem(j, k);
                        }
                    }
                }

                result += elem(0, i) * submatrix.det() * (i % 2 == 0 ? 1 : -1);
            }
        }

        return result;
    }

    Matrix Matrix::transpose() const 
    {
        Matrix result(m_cols, m_rows);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result.elem(j, i) = elem(i, j);
            }
        }

        return result;
    }

    Matrix Matrix::multiply(const Matrix& other) const
    {
        if (m_cols != other.m_rows)
            throw std::runtime_error("Left hand columns (" + std::to_string(m_cols) + ") do not match right hand rows (" + std::to_string(other.m_rows) + ")");

        Matrix result(m_rows, other.m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < other.m_cols; j++)
            {
                for (int k = 0; k < m_cols; k++)
                {
                    result.elem(i, j) += elem(i, k) * other.elem(k, j);
                }
            }
        }

        return result;
    }

    Matrix Matrix::sum_rows() const
    {
        Matrix result(m_rows, 1);

        for (int i = 0; i < m_rows; i++)
        {
            double sum = 0;

            for (int j = 0; j < m_cols; j++)
            {
                sum += elem(i, j);
            }

            result.elem(i, 0) = sum;
        }

        return result;
    }

    Matrix Matrix::sum_cols() const
    {
        Matrix result(1, m_cols);

        for (int i = 0; i < m_cols; i++)
        {
            double sum = 0;

            for (int j = 0; j < m_rows; j++)
            {
                sum += elem(i, j);
            }

            result.elem(0, i) = sum;
        }

        return result;
    }

    Matrix Matrix::square_elements() const
    {
        Matrix copy = *this;
        return copy * copy;
    }

    Matrix Matrix::log_elements() const
    {
        return unary_operation(UNARY(log));
    }

    Matrix Matrix::sigmoid() const
    {
        return unary_operation(UNARY(Math::sigmoid));
    }

    Matrix Matrix::sigmoid_derivative() const
    {
        return unary_operation(UNARY(Math::sigmoid_derivative));
    }

    Matrix Matrix::softmax() const
    {
        // TODO: Optimize?
        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            Math::softmax(row(i), result.row(i), m_cols);
        }

        return result;
    }

    Matrix Matrix::relu() const
    {
        return unary_operation(UNARY(Math::relu));
    }

    Matrix Matrix::relu_derivative() const
    {
        return unary_operation(UNARY(Math::relu_derivative));
    }

    Matrix Matrix::tanh() const
    {
        return unary_operation(UNARY(Math::tanh));
    }

    Matrix Matrix::tanh_derivative() const
    {
        return unary_operation(UNARY(Math::tanh_derivative));
    }

    Matrix Matrix::cross_entropy(const Matrix& other) const
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::runtime_error("Matrix dimensions do not match");

        Matrix result(m_rows, 1);

        for (int i = 0; i < m_rows; i++)
        {
            result.elem(i, 0) = Math::cross_entropy(row(i), other.row(i), m_cols);
        }

        return result;
    }

    double Matrix::cross_entropy_loss(const Matrix& other) const
    {
        Matrix result = cross_entropy(other).T();

        return std::accumulate(result[0], result[0] + m_rows, 0.0) / m_rows;
    }

    /*
     * Binary operators
     */

    Matrix Matrix::operator+(const Matrix& other) const
    {
        return elementwise_operation(other, LAMBDA_ADD);
    }

    Matrix Matrix::operator-(const Matrix& other) const
    {
        return elementwise_operation(other, LAMBDA_SUB);
    }

    Matrix Matrix::operator*(const Matrix& other) const
    {
        return elementwise_operation(other, LAMBDA_MUL);
    }

    Matrix Matrix::operator/(const Matrix& other) const
    {
        return elementwise_operation(other, LAMBDA_DIV);
    }

    /*
     * Augmented assignment operators
     */

    Matrix& Matrix::operator+=(const Matrix& other)
    {
        elementwise_operation(other, LAMBDA_ADD);
        return *this;
    }

    Matrix& Matrix::operator-=(const Matrix& other)
    {
        elementwise_operation(other, LAMBDA_SUB);
        return *this;
    }

    Matrix& Matrix::operator*=(const Matrix& other)
    {
        elementwise_operation(other, LAMBDA_MUL);
        return *this;
    }

    Matrix& Matrix::operator/=(const Matrix& other)
    {
        elementwise_operation(other, LAMBDA_DIV);
        return *this;
    }

    /*
     * Elementwise scalar operators
     */

    Matrix Matrix::operator+(double scalar) const
    {
        return binary_operation(scalar, LAMBDA_ADD);
    }

    Matrix Matrix::operator-(double scalar) const
    {
        return binary_operation(scalar, LAMBDA_SUB);
    }

    Matrix Matrix::operator*(double scalar) const
    {
        return binary_operation(scalar, LAMBDA_MUL);
    }

    Matrix Matrix::operator/(double scalar) const
    {
        return binary_operation(scalar, LAMBDA_DIV);
    }

    void Matrix::copy(const Matrix& other)
    {
        m_rows = other.m_rows;
        m_cols = other.m_cols;

        m_data = new double[m_rows * m_cols];
        std::copy(other.m_data, other.m_data + m_rows * m_cols, m_data);
    }

    void Matrix::move(Matrix& other)
    {
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_data = other.m_data;

        other.m_rows = 0;
        other.m_cols = 0;
        other.m_data = nullptr;
    }

    void Matrix::destroy()
    {
        if (m_data != nullptr)
            delete[] m_data;
        
        m_data = nullptr;
    }

    Matrix Matrix::unary_operation(std::function<double(double)> op) const
    {
        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result.elem(i, j) = op(elem(i, j));
            }
        }

        return result;
    }

    Matrix Matrix::binary_operation(double scalar, std::function<double(double, double)> op) const
    {
        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result.elem(i, j) = op(elem(i, j), scalar);
            }
        }

        return result;
    }

    void Matrix::elementwise_operation(const Matrix& other, std::function<double(double, double)> op)
    {
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                double elem1 = i >= m_rows || j >= m_cols ? 0.0 : elem(i, j);
                double elem2 = i >= other.m_rows || j >= other.m_cols ? 0.0 : other.elem(i, j);

                elem(i, j) = op(elem1, elem2);
            }
        }
    }

    Matrix Matrix::elementwise_operation(const Matrix& other, std::function<double(double, double)> op) const
    {
        int rows = std::max(m_rows, other.m_rows);
        int cols = std::max(m_cols, other.m_cols);

        Matrix result(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double elem1 = i >= m_rows || j >= m_cols ? 0.0 : elem(i, j);
                double elem2 = i >= other.m_rows || j >= other.m_cols ? 0.0 : other.elem(i, j);

                result.elem(i, j) = op(elem1, elem2);
            }
        }

        return result;
    }

    void Matrix::print() const
    {
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                std::cout << elem(i, j) << "\t";
            }
            std::cout << std::endl;
        }
    }
}
