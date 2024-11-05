#include <functional>

namespace Math
{
    class Matrix
    {
    public:
        static Matrix identity(int size);
        static Matrix I(int size) { return identity(size); }
        static Matrix cross(const Matrix& a, const Matrix& b);

        Matrix();
        Matrix(int rows, int cols = 1, double val = 0.0);
        Matrix(const Matrix& other);
        Matrix& operator=(const Matrix& other);
        Matrix(Matrix&& other) noexcept;
        Matrix& operator=(Matrix&& other) noexcept;

        ~Matrix();

        double* row(int row) const { return &m_data[row * m_cols]; }
        double* operator[](int row) { return this->row(row); }
        const double* operator[](int row) const { return this->row(row); }

        int rows() const { return m_rows; }
        int cols() const { return m_cols; }
        std::pair<int, int> shape() const { return { m_rows, m_cols }; }

        Matrix sigmoid() const;
        Matrix sigmoid_derivative() const;
        Matrix softmax() const;

        Matrix cross_entropy(const Matrix& other) const;
        double cross_entropy_loss(const Matrix& other) const;

        int determinant() const;
        int det() const { return determinant(); };

        Matrix transpose() const;
        Matrix T()         const { return transpose(); }

        Matrix square_elements() const;
        Matrix log_elements() const;

        Matrix sum_rows() const;
        Matrix sum_cols() const;

        Matrix multiply(const Matrix& other) const;
        
        bool operator==(const Matrix& other) const;

        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;
        Matrix operator/(const Matrix& other) const;

        Matrix& operator+=(const Matrix& other);
        Matrix& operator-=(const Matrix& other);
        Matrix& operator*=(const Matrix& other);
        Matrix& operator/=(const Matrix& other);

        Matrix operator+(double scalar) const;
        Matrix operator-(double scalar) const;
        Matrix operator*(double scalar) const;
        Matrix operator/(double scalar) const;

        // debug
        void print() const;

    private:
        int m_rows;
        int m_cols;

        double* m_data;

        double& elem(int row, int col) { return m_data[row * m_cols + col]; }
        double elem(int row, int col) const { return m_data[row * m_cols + col]; }

        Matrix unary_operation(std::function<double(double)> op) const;
        Matrix binary_operation(double scalar, std::function<double(double, double)> op) const;
        Matrix elementwise_operation(const Matrix& other, std::function<double(double, double)> op) const;
        void elementwise_operation(const Matrix& other, std::function<double(double, double)> op);

        void copy(const Matrix& other);
        void move(Matrix& other);
        void destroy();
    };
}
