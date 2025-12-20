#include "../include/relu.h"
ReLU::ReLU(int n_input, int d_input, int h_input, int w_input){
    this->n_input = n_input;
    this->h_input = h_input;
    this->h_output = h_input;
    input = new Tensor(n_input, d_input, h_input, w_input);
    output = new Tensor(n_input, d_input, h_input, w_input);
}
ReLU::~ReLU(){
    delete input;
    delete output;
}
Tensor* ReLU::forward(Tensor* input){
    assert(input->h == h_input && input->w == input->w);
    this->input->copy(input);
    for(int n = 0; n < this->input->n; ++n){
        for(int d = 0; d < this->input->d; ++d){
            this->input->matrix[n][d]->applyReLU(this->output->matrix[n][d]);
        }
    }
    return output;
}