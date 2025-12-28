#include "../include/flatten.h"
Flatten::Flatten(int n_input, int d_input, int h_input, int w_input){
    this->n_input = n_input;
    this->d_input = d_input;
    this->h_input = h_input;
    this->w_input = w_input;
    this->n_output = n_input;
    this->h_output = d_input * h_input * w_input;
    input = new Tensor(n_input, d_input, h_input, w_input);
    output = new Tensor(n_input, 1, d_input * h_input * w_input, 1);
}
Flatten::~Flatten(){
    delete input;
    delete output;
}
Tensor* Flatten::forward(Tensor* input){
    assert(input->h == h_input && input->w == w_input);
    this->input->copy(input);
    this->input->flattenWithTFOrder(output);
    return output;
}