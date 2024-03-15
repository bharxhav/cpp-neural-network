#ifndef UTILITYFUNCTIONS_H
#define UTILITYFUNCTIONS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Structure to represent a matrix
struct Mat
{
    int rows;
    int cols;
    std::vector<std::vector<double> > data;

    Mat(int rows, int cols);

    void randomize();
};

// Utility function to print a matrix
void printMatrix(const Mat &mat);

// Utility function to perform element-wise sigmoid activation on a matrix
Mat sigmoid(const Mat &mat);

// Utility function to perform element-wise ReLU activation on a matrix
Mat relu(const Mat &mat);

// Utility function to calculate the transpose of a matrix
Mat transpose(const Mat &mat);

// Utility function to perform element-wise multiplication of two matrices
Mat hadamardProduct(const Mat &mat1, const Mat &mat2);

// Utility function to perform matrix multiplication
Mat dotProduct(const Mat &mat1, const Mat &mat2);

#endif
