#include "../include/pooling.h"
Pooling::Pooling(int n_input, int d_input, int h_input, int w_input, int poolSize, int stride){
    this->n_input = n_input;
    this->d_input = d_input;
    this->h_input = h_input;
    this->w_input = w_input;
    this->poolSize = poolSize;
    this->stride = stride;

    this->n_output = n_input;
    this->d_output = d_input;
    this->h_output = (h_input - poolSize) / stride + 1;
    this->w_output = (w_input - poolSize) / stride + 1;

    input = new Tensor(n_input, d_input, h_input, w_input);
    output = new Tensor(n_output, d_output, h_output, w_output);
}
Pooling::~Pooling(){
    delete input;
    delete output;
}
Tensor* Pooling::forward(Tensor* input){
    assert(input->h == h_input && input->w == w_input);
    this->input->copy(input);
    this->input->maxPool(output, poolSize, stride);
    return output;
}