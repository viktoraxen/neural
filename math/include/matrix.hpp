#include <vector>
#include <functional>

namespace Math
{
    class Matrix
    {
    public:
        static Matrix identity(int size);
        static Matrix cross(const Matrix& a, const Matrix& b);

        Matrix();
        Matrix(int rows, int cols = 1, double val = 0.0);
        Matrix(const Matrix& other);
        Matrix& operator=(const Matrix& other);

        ~Matrix();

        double* operator[](int row) { return m_data[row]; }
        const double* operator[](int row) const { return m_data[row]; }

        int rows() const { return m_rows; }
        int cols() const { return m_cols; }

        Matrix sigmoid() const;
        Matrix softmax() const;

        Matrix cross_entropy(const Matrix& other) const;

        int determinant() const;
        int det() const { return determinant(); };

        Matrix transpose() const;
        Matrix T()         const { return transpose(); }

        Matrix multiply(const Matrix& other) const;
        
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

    private:
        int m_rows;
        int m_cols;

        double** m_data;

        Matrix scalar_operation(double scalar, std::function<double(double, double)> op) const;
        Matrix elementwise_operation(const Matrix& other, std::function<double(double, double)> op) const;

        void copy(const Matrix& other);
        void destroy();
    };
}
