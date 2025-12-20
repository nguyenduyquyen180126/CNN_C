#ifndef FLATTEN_H
#define FLATTEN_H
#include "tensor.h"
#include "utils.h"
class Flatten{
public:
    Tensor* input;
    Tensor* output;
    int n_input, d_input, h_input, w_input;
    int n_output, h_output;
    Flatten(int n_input, int d_input, int h_input, int w_input);
    ~Flatten();
    Tensor* forward(Tensor* input);
};
#endif