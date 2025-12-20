#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "tensor.h"
#include <iostream>
#include <cmath>
#include "utils.h"
class Softmax{
public:
    // Tensor *input;
    Tensor* output;
    Matrix* weights;
    // Tensor *z;
    Matrix* bias;
    Matrix* dL_dB;
    Matrix* dL_dW;
    Matrix* dL_dX;
    int n_input;
    int h_input;
    int h_output;
    Softmax(int n_input, int d_input, int h_input, int w_input, int h_output);
    ~Softmax();
    Tensor* forward(Tensor* input);
    Matrix* backward(Matrix* d_Out);
    void printParam();
    void readWeightsFromFile(const char* filename);
    void readBiasFromFile(const char* filename);
};
#endif