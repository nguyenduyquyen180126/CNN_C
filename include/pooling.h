#ifndef POOLING_H
#define POOLING_H
#include "tensor.h"
#include "utils.h"
class Pooling{
public:
    Tensor* input;
    Tensor* output;
    int n_input, d_input, h_input, w_input;
    int n_output, d_output, h_output, w_output;
    int poolSize;
    int stride;
    Pooling(int n_input, int d_input, int h_input, int w_input, int poolSize, int stride);
    ~Pooling();
    Tensor* forward(Tensor* input);
};
#endif