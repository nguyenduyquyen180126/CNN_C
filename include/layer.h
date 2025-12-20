#ifndef LAYER_H
#define LAYER_H

#include "tensor.h"
#include "utils.h"
#include "dense.h"
#include "softmax.h"
#include "relu.h"
#include "pooling.h"
#include "flatten.h"
#include "conv.h"
#include "batch_norm.h"

/*
To create a layer, specify the type and necessary parameters.
Convolutional Layer:
Layer* convLayer = new Layer(n_input, d_input, h_input, w_input, n_kernels, kernel_size, stride, padding, convLayer);
Relu Layer:
Layer* reluLayer = new Layer(n_input, d_input, h_input, w_input, reluLayer);
Pooling Layer:
Layer* poolingLayer = new Layer(n_input, d_input, h_input, w_input, poolSize, stride, poolingLayer);
Flatten Layer:
Layer* flattenLayer = new Layer(n_input, d_input, h_input, w_input, flattenLayer);
Dense Layer:
Layer* denseLayer = new Layer(n_input, d_input, h_input, w_input, h_output, denseLayer);
Softmax Layer:
Layer* softmaxLayer = new Layer(n_input, d_input, h_input, w_input, h_output, softmaxLayer);
*/

enum LayerType{
    denseLayer,
    softmaxLayer,
    convLayer,
    reluLayer,
    poolingLayer,
    flattenLayer,
    batchNormLayer
};

class Layer{
public:
    Dense* dense = nullptr;
    Softmax* softmax = nullptr;
    ReLU* relu = nullptr;
    Conv* conv = nullptr;
    Pooling* pooling = nullptr;
    Flatten* flatten = nullptr;
    BatchNorm* batchNorm = nullptr;
    
    Layer(int n_input, int d_input, int h_input, int w_input, int n_kernels, int kernel_size, int stride, int padding, LayerType type);
    Layer(int n_input, int d_input, int h_input, int w_input, LayerType type);
    Layer(int n_input, int d_input, int h_input, int w_input, int poolSize, int stride, LayerType type);
    Layer(int n_input, int d_input, int h_input, int w_input, int h_output, LayerType type);
    Layer(int n_input, int d_input, int h_input, int w_input, LayerType type, std::string bn_type);
    ~Layer();
};

#endif