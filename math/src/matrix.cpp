#include "matrix.hpp"

#include "math.hpp"

#include <algorithm>
#include <iostream>
#include <cassert>

#define LAMBDA_ADD [](double a, double b) { return a + b; }
#define LAMBDA_SUB [](double a, double b) { return a - b; }
#define LAMBDA_MUL [](double a, double b) { return a * b; }
#define LAMBDA_DIV [](double a, double b) { return a / b; }

namespace Math
{
    Matrix::Matrix()
        : m_rows(0)
        , m_cols(0)
    {}

    // Constructor
    Matrix::Matrix(int rows, int cols, double val)
        : m_rows(rows)
        , m_cols(cols)
    {
        m_data = new double*[rows];

        for (int i = 0; i < rows; i++)
        {
            m_data[i] = new double[cols];

            for (int j = 0; j < cols; j++)
            {
                m_data[i][j] = val;
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
                if (m_data[i][j] != other.m_data[i][j])
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
            result = m_data[0][0];
        }
        else if (m_rows == 2)
        {
            result = m_data[0][0] * m_data[1][1] - m_data[0][1] * m_data[1][0];
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
                            submatrix[j - 1][k] = m_data[j][k];
                        }
                        else if (k > i)
                        {
                            submatrix[j - 1][k - 1] = m_data[j][k];
                        }
                    }
                }

                result += m_data[0][i] * submatrix.det() * (i % 2 == 0 ? 1 : -1);
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
                result.m_data[j][i] = m_data[i][j];
            }
        }

        return result;
    }

    Matrix Matrix::multiply(const Matrix& other) const
    {
        if (m_cols != other.m_rows)
            throw std::runtime_error("Matrix dimensions do not match");

        Matrix result(m_rows, other.m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < other.m_cols; j++)
            {
                for (int k = 0; k < m_cols; k++)
                {
                    result.m_data[i][j] += m_data[i][k] * other.m_data[k][j];
                }
            }
        }

        return result;
    }

    Matrix Matrix::sigmoid() const
    {
        return scalar_operation(0, [](double a, double _) { return Math::sigmoid(a); });
    }

    Matrix Matrix::softmax() const
    {
        // TODO: Optimize?
        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            Math::softmax(m_data[i], result[i], m_cols);
        }

        return result;
    }

    Matrix Matrix::cross_entropy(const Matrix& other) const
    {
        if (m_rows != other.m_rows || m_cols != other.m_cols)
            throw std::runtime_error("Matrix dimensions do not match");

        Matrix result(m_rows, 1);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result[i][j] = Math::cross_entropy(m_data[i], other.m_data[i], m_cols);
            }
        }

        return result;
    }

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

    Matrix Matrix::operator+(double scalar) const
    {
        return scalar_operation(scalar, LAMBDA_ADD);
    }

    Matrix Matrix::operator-(double scalar) const
    {
        return scalar_operation(scalar, LAMBDA_SUB);
    }

    Matrix Matrix::operator*(double scalar) const
    {
        return scalar_operation(scalar, LAMBDA_MUL);
    }

    Matrix Matrix::operator/(double scalar) const
    {
        return scalar_operation(scalar, LAMBDA_DIV);
    }

    void Matrix::copy(const Matrix& other)
    {
        m_rows = other.m_rows;
        m_cols = other.m_cols;

        m_data = new double*[m_rows];
        
        for (int i = 0; i < m_rows; i++)
        {
            m_data[i] = new double[m_cols];

            for (int j = 0; j < m_cols; j++)
            {
                m_data[i][j] = other.m_data[i][j];
            }
        }
    }

    void Matrix::destroy()
    {
        for (int i = 0; i < m_rows; i++)
        {
            if (m_data[i] != nullptr)
                delete[] m_data[i];
        }

        if (m_data != nullptr)
            delete[] m_data;
    }

    Matrix Matrix::scalar_operation(double scalar, std::function<double(double, double)> op) const
    {
        Matrix result(m_rows, m_cols);

        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                result.m_data[i][j] = op(m_data[i][j], scalar);
            }
        }

        return result;
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
                double elem1 = i >= m_rows || j >= m_cols ? 0.0 : m_data[i][j];
                double elem2 = i >= other.m_rows || j >= other.m_cols ? 0.0 : other.m_data[i][j];

                result.m_data[i][j] = op(elem1, elem2);
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
                std::cout << m_data[i][j] << "\t";
            }
            std::cout << std::endl;
        }
    }

    Matrix Matrix::identity(int size)
    {
        Matrix result(size, size);

        for (int i = 0; i < size; i++)
        {
            result[i][i] = 1;
        }

        return result;
    }

    Matrix Matrix::cross(const Matrix& a, const Matrix& b)
    {
        if (a.cols() != b.rows())
            throw std::runtime_error("Matrix dimensions do not match");

        return a.T().multiply(b);
    }
}
