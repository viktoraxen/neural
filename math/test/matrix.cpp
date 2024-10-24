#include "math.hpp"
#include <gtest/gtest.h>

using namespace Math;

TEST(Matrix, Initialization)
{
    Matrix mat(2, 2);

    EXPECT_EQ(mat[0][0], 0);
    EXPECT_EQ(mat[1][0], 0);
    EXPECT_EQ(mat[0][1], 0);
    EXPECT_EQ(mat[1][1], 0);
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 2);
}

TEST(Matrix, ElementAssignment)
{
    Matrix mat(2, 2);

    mat[0][0] = 1;
    mat[1][0] = 2;
    mat[0][1] = 3;
    mat[1][1] = 4;

    EXPECT_EQ(mat[0][0], 1);
    EXPECT_EQ(mat[1][0], 2);
    EXPECT_EQ(mat[0][1], 3);
    EXPECT_EQ(mat[1][1], 4);
}

TEST(Matrix, Assignment)
{
    Matrix mat(2, 2);
    Matrix mat2 = mat;

    EXPECT_EQ(mat, mat2);
}

TEST(Matrix, CopyConstructor)
{
    Matrix mat(2, 2);
    Matrix mat2(mat);

    EXPECT_EQ(mat, mat2);
}

TEST(Matrix, Elementwise)
{
    Matrix mat(2, 2);
    Matrix mat2(2, 2);

    mat[0][0] = 1;
    mat[1][0] = 2;
    mat[0][1] = 3;
    mat[1][1] = 4;

    mat2[0][0] = 5;
    mat2[1][0] = 6;
    mat2[0][1] = 7;
    mat2[1][1] = 8;

    Matrix mat3 = mat + mat2;

    EXPECT_EQ(mat3[0][0], mat[0][0] + mat2[0][0]);
    EXPECT_EQ(mat3[1][0], mat[1][0] + mat2[1][0]);
    EXPECT_EQ(mat3[0][1], mat[0][1] + mat2[0][1]);
    EXPECT_EQ(mat3[1][1], mat[1][1] + mat2[1][1]);

    Matrix mat4 = mat - mat2;

    EXPECT_EQ(mat4[0][0], mat[0][0] - mat2[0][0]);
    EXPECT_EQ(mat4[1][0], mat[1][0] - mat2[1][0]);
    EXPECT_EQ(mat4[0][1], mat[0][1] - mat2[0][1]);
    EXPECT_EQ(mat4[1][1], mat[1][1] - mat2[1][1]);

    Matrix mat5 = mat * mat2;

    EXPECT_EQ(mat5[0][0], mat[0][0] * mat2[0][0]);
    EXPECT_EQ(mat5[1][0], mat[1][0] * mat2[1][0]);
    EXPECT_EQ(mat5[0][1], mat[0][1] * mat2[0][1]);
    EXPECT_EQ(mat5[1][1], mat[1][1] * mat2[1][1]);

    Matrix mat6 = mat / mat2;

    EXPECT_EQ(mat6[0][0], mat[0][0] / mat2[0][0]);
    EXPECT_EQ(mat6[1][0], mat[1][0] / mat2[1][0]);
    EXPECT_EQ(mat6[0][1], mat[0][1] / mat2[0][1]);
    EXPECT_EQ(mat6[1][1], mat[1][1] / mat2[1][1]);
}

TEST(Matrix, ElementwiseUnmatchingDimensions)
{
    Matrix mat(2, 2);
    Matrix mat2(2, 3);
    Matrix mat3 = mat + mat2;

    EXPECT_EQ(mat3.rows(), 2);
    EXPECT_EQ(mat3.cols(), 3);
}

TEST(Matrix, MatrixMultiplication)
{
    Matrix mat1(2, 4);
    Matrix mat2(4, 1);

    mat1 = mat1 + 3;
    mat2 = mat2 + 2.5;

    Matrix mat3 = mat1.multiply(mat2);

    EXPECT_EQ(mat3[0][0], mat1[0][0] * mat2[0][0] + mat1[0][1] * mat2[1][0] + mat1[0][2] * mat2[2][0] + mat1[0][3] * mat2[3][0]);
    EXPECT_EQ(mat3.rows(), mat1.rows());
    EXPECT_EQ(mat3.cols(), mat2.cols());
}

TEST(Matrix, MatrixMultiplicationUnmatchingDimensions)
{
    Matrix mat1(2, 4);
    Matrix mat2(3, 1);

    EXPECT_THROW(mat1.multiply(mat2), std::runtime_error);
}

TEST(Matrix, Transpose)
{
    Matrix mat(2, 3);
    mat[0][0] = 1;
    mat[1][0] = 2;
    mat[0][1] = 3;
    mat[1][1] = 4;
    mat[0][2] = 5;
    mat[1][2] = 6;

    // Main functionality
    Matrix mat2 = mat.transpose();

    EXPECT_EQ(mat2[0][0], mat[0][0]);
    EXPECT_EQ(mat2[0][1], mat[1][0]);
    EXPECT_EQ(mat2[1][0], mat[0][1]);
    EXPECT_EQ(mat2[1][1], mat[1][1]);
    EXPECT_EQ(mat2[2][0], mat[0][2]);
    EXPECT_EQ(mat2[2][1], mat[1][2]);

    // Alias
    Matrix mat3 = mat.T();

    EXPECT_EQ(mat2, mat3);

    // Reverse
    Matrix mat4 = mat2.T();

    EXPECT_EQ(mat4, mat);
}

TEST(Matrix, Sigmoid)
{
    Matrix mat(2, 2);

    mat[0][0] = 1;
    mat[1][0] = 2;
    mat[0][1] = 3;
    mat[1][1] = 4;

    Matrix sig = mat.sigmoid();

    EXPECT_EQ(sig[0][0], 1 / (1 + exp(-1)));
    EXPECT_EQ(sig[1][0], 1 / (1 + exp(-2)));
    EXPECT_EQ(sig[0][1], 1 / (1 + exp(-3)));
    EXPECT_EQ(sig[1][1], 1 / (1 + exp(-4)));
}

TEST(Matrix, Determinant)
{
    Matrix mat(2, 2);

    mat[0][0] = 1;
    mat[1][0] = 2;
    mat[0][1] = 3;
    mat[1][1] = 4;

    EXPECT_EQ(mat.det(), -2);

    Matrix mat2(3, 3);

    mat2[0][0] = 1;
    mat2[1][0] = 2;
    mat2[2][0] = 3;
    mat2[0][1] = 4;
    mat2[1][1] = 5;
    mat2[2][1] = 6;
    mat2[0][2] = 7;
    mat2[1][2] = 8;
    mat2[2][2] = 9;

    EXPECT_EQ(mat2.det(), 0);

    Matrix mat3(4, 4);

    mat3[0][0] = 1;
    mat3[0][1] = 0;
    mat3[0][2] = 6;
    mat3[0][3] = 7;
    mat3[1][0] = 2;
    mat3[1][1] = 7;
    mat3[1][2] = 8;
    mat3[1][3] = 1;
    mat3[2][0] = 2;
    mat3[2][1] = 3;
    mat3[2][2] = 9;
    mat3[2][3] = 8;
    mat3[3][0] = 1;
    mat3[3][1] = 4;
    mat3[3][2] = 5;
    mat3[3][3] = 7;

    EXPECT_EQ(mat3.det(), -63);
}

TEST(Matrix, Identity)
{
    Matrix mat = Matrix::identity(3);

    EXPECT_EQ(mat[0][0], 1);
    EXPECT_EQ(mat[1][1], 1);
    EXPECT_EQ(mat[2][2], 1);
    EXPECT_EQ(mat[0][1], 0);
    EXPECT_EQ(mat[0][2], 0);
    EXPECT_EQ(mat[1][0], 0);
    EXPECT_EQ(mat[1][2], 0);
    EXPECT_EQ(mat[2][0], 0);
    EXPECT_EQ(mat[2][1], 0);
}

TEST(Matrix, Stack)
{
    Matrix row(1, 5);

    row[0][0] = 1;
    row[0][1] = 2;
    row[0][2] = 3;
    row[0][3] = 4;
    row[0][4] = 5;

    Matrix mat = Matrix::stack(row, 3);

    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 5);

    for (int i = 0; i < 5; i++)
    {
        EXPECT_EQ(mat[0][i], i + 1);
        EXPECT_EQ(mat[1][i], i + 1);
        EXPECT_EQ(mat[2][i], i + 1);
    }
}
