#ifndef DENSE_H
#define DENSE_H
#include "tensor.h"
#include "dense.h"
#include "utils.h"
class Dense{
public:
    Matrix* weights;
    Matrix* bias;
    Tensor* input;
    Tensor* output;
    Tensor* dL_dW;
    int n_input, d_input = 1 , h_input, w_input = 1;
    int n_output, d_output = 1, h_output, w_output = 1;
    Dense(int n_input, int d_input, int h_input, int w_input, int h_output);
    ~Dense();
    Tensor *forward(Tensor* input);
    Matrix *backward(Matrix* d_Out, double lr);
    void printParam();
    void readWeightsFromFile(const char* filename);
    void readBiasFromFile(const char* filename);
};
#endif