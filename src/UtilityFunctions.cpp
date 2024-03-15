#include "UtilityFunctions.h"

// Constructor for Mat structure
Mat::Mat(int rows, int cols) : rows(rows), cols(cols)
{
    data.resize(rows, std::vector<double>(cols, 0.0));
}

// Function to randomize the values of a matrix
void Mat::randomize()
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            data[i][j] = (rand() / double(RAND_MAX)) * 2 - 1; // Random value between -1 and 1
        }
    }
}

// Function to print a matrix
void printMatrix(const Mat &mat)
{
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            std::cout << mat.data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// Sigmoid activation function
Mat sigmoid(const Mat &mat)
{
    Mat result(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            result.data[i][j] = 1.0 / (1.0 + exp(-mat.data[i][j]));
        }
    }
    return result;
}

// ReLU activation function
Mat relu(const Mat &mat)
{
    Mat result(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            result.data[i][j] = std::max(0.0, mat.data[i][j]);
        }
    }
    return result;
}

// Transpose of a matrix
Mat transpose(const Mat &mat)
{
    Mat result(mat.cols, mat.rows);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            result.data[j][i] = mat.data[i][j];
        }
    }
    return result;
}

// Element-wise multiplication of two matrices (Hadamard product)
Mat hadamardProduct(const Mat &mat1, const Mat &mat2)
{
    Mat result(mat1.rows, mat1.cols);
    for (int i = 0; i < mat1.rows; ++i)
    {
        for (int j = 0; j < mat1.cols; ++j)
        {
            result.data[i][j] = mat1.data[i][j] * mat2.data[i][j];
        }
    }
    return result;
}

// Matrix multiplication
Mat dotProduct(const Mat &mat1, const Mat &mat2)
{
    Mat result(mat1.rows, mat2.cols);
    for (int i = 0; i < mat1.rows; ++i)
    {
        for (int j = 0; j < mat2.cols; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < mat1.cols; ++k)
            {
                sum += mat1.data[i][k] * mat2.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
}
