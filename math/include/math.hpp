#ifndef MATH_HPP
#define MATH_HPP

#include <cmath>
#include <vector>
#include <functional>

namespace Math
{
    static double sigmoid(double a) { return 1 / (1 + exp(-a)); }

    class Matrix
    {
    public:
        static Matrix identity(int size);
        static Matrix stack(const Matrix& a, int rows);

        Matrix();
        Matrix(int rows, int cols);
        Matrix(const Matrix& other);
        Matrix& operator=(const Matrix& other);

        ~Matrix() = default;

        std::vector<double>& operator[](int row) { return m_data[row]; }
        const std::vector<double>& operator[](int row) const { return m_data[row]; }

        int rows() const { return m_rows; }
        int cols() const { return m_cols; }

        int determinant() const;
        int det() const { return determinant(); };

        Matrix transpose() const;
        Matrix T()         const { return transpose(); }

        Matrix multiply(const Matrix& other) const;
        
        Matrix dot(const Matrix& other) const;
        Matrix cross(const Matrix& other) const;

        Matrix sigmoid() const;

        bool operator==(const Matrix& other) const;

        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator/(const Matrix& other) const;

        Matrix operator+(double scalar) const;
        Matrix operator-(double scalar) const;
        Matrix operator*(double scalar) const;
        Matrix operator/(double scalar) const;

        // DEBUG
        void print() const;
        
        static Matrix filled(int rows, int cols, double value);

    private:
        int m_rows;
        int m_cols;

        std::vector<std::vector<double>> m_data;

        Matrix scalar_operation(double scalar, std::function<double(double, double)> op) const;
        Matrix elementwise_operation(const Matrix& other, std::function<double(double, double)> op) const;
        void copy(const Matrix& other);
    };
}

#endif // MATH_HPP
