#include "../include/layer.h"
Layer::Layer(int n_input, int d_input, int h_input, int w_input, int n_kernels, int kernel_size, int stride, int padding, LayerType type){
    conv = new Conv(n_input, d_input, h_input, w_input, n_kernels, kernel_size, stride, padding);
}
Layer::Layer(int n_input, int d_input, int h_input, int w_input, int poolSize, int stride, LayerType type){
    if(type == poolingLayer){
        pooling = new Pooling(n_input, d_input, h_input, w_input, poolSize, stride);
    }
}
Layer::Layer(int n_input, int d_input, int h_input, int w_input, int h_output, LayerType type){
    if(type == denseLayer){
        dense = new Dense(n_input, d_input, h_input, w_input, h_output);
    }
    else if(type == softmaxLayer){
        softmax = new Softmax(n_input, d_input, h_input, w_input, h_output);
    }
}
Layer::Layer(int n_input, int d_input, int h_input, int w_input, LayerType type){
    if(type == reluLayer){
        relu = new ReLU(n_input, d_input, h_input, w_input);
    }
    else if(type == flattenLayer){ 
        flatten = new Flatten(n_input, d_input, h_input, w_input); 
    }
}
Layer::Layer(int n_input, int d_input, int h_input, int w_input, LayerType type, std::string bn_type){
    if(type == batchNormLayer){
        batchNorm = new BatchNorm(n_input, d_input, h_input, w_input, 0.9, bn_type);
    }
}
Layer::~Layer(){
    if(dense){
        delete dense;
    }
    if(softmax){
        delete softmax;
    }
    if(relu){
        delete relu;
    }
    if(conv){
        delete conv;
    }
    if(pooling){
        delete pooling;
    }
    if(flatten){
        delete flatten;
    }
}