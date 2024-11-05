#include <matrix.hpp>
#include <gtest/gtest.h>

#include <cmath>

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

TEST(Matrix, AugmentedAssignment)
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

    mat += mat2;

    EXPECT_EQ(mat[0][0], 1 + 5);
    EXPECT_EQ(mat[1][0], 2 + 6);
    EXPECT_EQ(mat[0][1], 3 + 7);
    EXPECT_EQ(mat[1][1], 4 + 8);

    mat -= mat2;

    EXPECT_EQ(mat[0][0], 1);
    EXPECT_EQ(mat[1][0], 2);
    EXPECT_EQ(mat[0][1], 3);
    EXPECT_EQ(mat[1][1], 4);

    mat *= mat2;

    EXPECT_EQ(mat[0][0], 1 * 5);
    EXPECT_EQ(mat[1][0], 2 * 6);
    EXPECT_EQ(mat[0][1], 3 * 7);
    EXPECT_EQ(mat[1][1], 4 * 8);

    mat /= mat2;

    EXPECT_EQ(mat[0][0], 1);
    EXPECT_EQ(mat[1][0], 2);
    EXPECT_EQ(mat[0][1], 3);
    EXPECT_EQ(mat[1][1], 4);
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

TEST(Matrix, Softmax)
{
    Matrix mat(3, 3);

    mat[0][0] = 1;
    mat[0][1] = 2;
    mat[0][2] = 3;
    mat[1][0] = 4;
    mat[1][1] = 5;
    mat[1][2] = 6;
    mat[2][0] = 7;
    mat[2][1] = 8;
    mat[2][2] = 9;

    Matrix sm = mat.softmax();

    double E1 = exp(1);
    double E2 = exp(2);
    double E3 = exp(3);
    double E_sum = E1 + E2 + E3;

    EXPECT_EQ(sm[0][0], E1 / E_sum);
    EXPECT_EQ(sm[0][1], E2 / E_sum);
    EXPECT_EQ(sm[0][2], E3 / E_sum);

    E1 = exp(4);
    E2 = exp(5);
    E3 = exp(6);
    E_sum = E1 + E2 + E3;

    EXPECT_EQ(sm[1][0], E1 / E_sum);
    EXPECT_EQ(sm[1][1], E2 / E_sum);
    EXPECT_EQ(sm[1][2], E3 / E_sum);

    E1 = exp(7);
    E2 = exp(8);
    E3 = exp(9);
    E_sum = E1 + E2 + E3;

    EXPECT_EQ(sm[2][0], E1 / E_sum);
    EXPECT_EQ(sm[2][1], E2 / E_sum);
    EXPECT_EQ(sm[2][2], E3 / E_sum);
}

TEST(Matrix, SumRows)
{
    Matrix mat(2, 3);

    mat[0][0] = 1;
    mat[0][1] = 2;
    mat[0][2] = 3;
    mat[1][0] = 4;
    mat[1][1] = 5;
    mat[1][2] = 6;

    Matrix sum = mat.sum_rows();

    EXPECT_EQ(sum[0][0], 1 + 2 + 3);
    EXPECT_EQ(sum[1][0], 4 + 5 + 6);
}

TEST(Matrix, SquareElements)
{
    Matrix mat(2, 2);

    mat[0][0] = 1;
    mat[1][0] = 2;
    mat[0][1] = 3;
    mat[1][1] = 4;

    Matrix sq = mat.square_elements();

    EXPECT_EQ(sq[0][0], 1);
    EXPECT_EQ(sq[1][0], 4);
    EXPECT_EQ(sq[0][1], 9);
    EXPECT_EQ(sq[1][1], 16);
}

TEST(Matrix, LogElements)
{
    Matrix mat(2, 2);

    mat[0][0] = 1;
    mat[1][0] = 2;
    mat[0][1] = 3;
    mat[1][1] = 4;

    Matrix log_result = mat.log_elements();

    EXPECT_EQ(log_result[0][0], log(1));
    EXPECT_EQ(log_result[1][0], log(2));
    EXPECT_EQ(log_result[0][1], log(3));
    EXPECT_EQ(log_result[1][1], log(4));
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

// TEST(Matrix, CrossEntropy)
// {
//     Matrix mat(2, 3);
//     Matrix mat2(2, 3);
//
//     mat[0][0] = 0;
//     mat[0][1] = 1;
//     mat[0][2] = 0;
//
//     mat[1][0] = 1;
//     mat[1][1] = 0;
//     mat[1][2] = 0;
//
//     mat2[0][0] = 0.2;
//     mat2[0][1] = 0.7;
//     mat2[0][2] = 0.1;
//
//     mat2[1][0] = 0.2;
//     mat2[1][1] = 0.7;
//     mat2[1][2] = 0.1;
//
//     Matrix res(2, 1);
//     res[0][0] = -log(0.7);
//     res[1][0] = -log(0.2);
//
//     EXPECT_EQ(mat.cross_entropy(mat2), res);
// }

// TEST(Matrix, CrossEntropyLoss)
// {
//     Matrix mat(1, 3);
//     Matrix mat2(1, 3);
//
//     mat[0][0] = 0;
//     mat[0][1] = 1;
//     mat[0][2] = 0;
//
//     mat2[0][0] = 0.2;
//     mat2[0][1] = 0.7;
//     mat2[0][2] = 0.1;
//
//     EXPECT_EQ(mat.cross_entropy_loss(mat2), -log(0.7));
// }

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
