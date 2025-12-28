#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include <vector>
#include <cassert>
#include <stdlib.h>
#include "utils.h"
class Matrix{
public:
    double** data;
    int rows;
    int cols;
    Matrix(int r, int c, bool random=false);
    Matrix() = default;
    ~Matrix();
    void print();
    void assign(double val);
    // void assignRandom(double mu, double sigma);
    void add(Matrix* other, Matrix* result);
    void multiply(Matrix* other, Matrix* result);
    void addNumber(double num, Matrix* result);
    void applyReLU(Matrix* result);
    void convolve(Matrix* kernel, Matrix* result, int stride, int padding);
    void maxPool(Matrix* result, int poolSize, int stride);
    void flatten(Matrix* result);
    void padding(Matrix* result, int pad);
    void copy(Matrix* src);
    void transpose(Matrix* result);
    void subtract(Matrix* other, Matrix* result);
};

class Tensor{
public:
    Matrix*** matrix;
    int n, d, h, w;
    Tensor(int n, int d, int h, int w, bool random=false);
    ~Tensor();
    void print();
    void convolve(Tensor* kernel, Tensor* result, int stride, int padding);
    void maxPool(Tensor* result, int poolSize, int stride);
    void flatten(Tensor* result);
    void copy(Tensor* src);
    void assignMatrix(Matrix* mat, int batchIndex, int depthIndex);
    void subtract(Tensor* other, Tensor* result);
    void mergeTensorToMatrix(Matrix* result);
    void readFromFile(const char * filename);
    void writeToFile(const char * filename);
    void flattenWithTFOrder(Tensor *result);
};

#endif