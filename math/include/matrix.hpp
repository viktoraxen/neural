#include <vector>
#include <functional>

namespace Math
{
    class Matrix
    {
    public:
        static Matrix identity(int size);
        static Matrix filled(int rows, int cols, double value);
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

        Matrix sigmoid() const;

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

        std::vector<std::vector<double>> m_data;

        Matrix scalar_operation(double scalar, std::function<double(double, double)> op) const;
        Matrix elementwise_operation(const Matrix& other, std::function<double(double, double)> op) const;

        void copy(const Matrix& other);
    };
}
