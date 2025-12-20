#ifndef RELU_H
#define RELU_H
#include "tensor.h"
#include <algorithm>
#include "utils.h"
class ReLU{
public:
    Tensor* input;
    Tensor* output;
    int n_input, d_input, h_input, w_input;
    int h_output;
    ReLU(int n_input, int d_input, int h_input, int w_input);
    ~ReLU();
    Tensor* forward(Tensor* input);
};
#endif