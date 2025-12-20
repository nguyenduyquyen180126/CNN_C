#ifndef MODEL_H
#define MODEL_H
#include <vector>
#include "layer.h" 
#include "tensor.h"
#include <math.h>
#include "dataloader.h"
class Model{
public:
    std::vector<Layer*> model_sequence;
    int epochs;
    int n_input, d_input, h_input, w_input;
    Matrix* batch_costs;
    Tensor* output = nullptr;
    Model();
    ~Model();
    virtual void calLoss(Tensor* labels);
    void addLayer(Layer* layer);
    virtual Tensor* forward(Tensor* input);
};
#endif