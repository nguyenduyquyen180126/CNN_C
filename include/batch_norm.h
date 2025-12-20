#ifndef BATCH_NORM_H
#define BATCH_NORM_H
#include "tensor.h"
#include <vector>
class BatchNorm {
public:
    int n, d, h, w;
    Matrix* gamma;
    Matrix* beta; 
    Matrix* running_mean;
    Matrix* running_var;
    Tensor* output;
    Tensor* input;
    double momentum;
    double epsilon;
    BatchNorm(int n, int d, int h, int w, double momentum, std::string type);
    void printParam();
    ~BatchNorm();
    Tensor* inference_forward(Tensor* input);
    void readParamFromFile(const char* gamma_file, const char* beta_file, const char* running_mean_file, const char* running_var_file);
};
#endif