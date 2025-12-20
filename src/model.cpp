#include "../include/model.h"
Model::Model(){
    this->epochs = 100;
}
Model::~Model(){
    for(int i=0; i<model_sequence.size(); i++){
        delete model_sequence[i];
    }
}
void Model::addLayer(Layer* layer){
    model_sequence.push_back(layer);
}
Tensor* Model::forward(Tensor* input){
    // Virtual function
    std::cout << "Forward pass:" << std::endl;
    return nullptr;
}
void Model::calLoss(Tensor* labels){
    // Virtual function
}