#ifndef CONV_H
#define CONV_H
#include "tensor.h"
#include "utils.h"
#include <vector>
class Conv{
public:
    Tensor* input;
    Tensor* output;
    Tensor* kernels;
    Matrix* bias;
    int n_kernels;
    int kernel_size;
    int stride;
    int padding;
    int n_input, d_input, h_input, w_input;
    int n_output, d_output, h_output, w_output;
    Conv(int n_input, int d_input, int h_input, int w_input, int n_kernels, int kernel_size, int stride, int padding);
    ~Conv();
    Tensor* forward(Tensor* input);
    void printParam();
    void readKernelsFromFile(const char* filename);
    void readBiasFromFile(const char* filename);
};
#endif