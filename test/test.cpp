#include "model.h"
#include "dataloader.h"
#include "conv.h"
#include <stdlib.h>

class CNN : public Model{
public:
    CNN() : Model() {

        addLayer(new Layer(1, 3, 32, 32, 32, 3, 1, 1, convLayer)); // Conv Layer
        addLayer(new Layer(1, 32, 32, 32, batchNormLayer, "conv")); // BatchNorm Layer
        addLayer(new Layer(1, 32, 32, 32, reluLayer)); // ReLU Layer
        addLayer(new Layer(1, 32, 32, 32, 2, 2, poolingLayer)); // Pooling Layer

        addLayer(new Layer(1, 32, 16, 16, 64, 3, 1, 1, convLayer)); // Conv Layer
        addLayer(new Layer(1, 64, 16, 16, batchNormLayer, "conv")); // BatchNorm Layer
        addLayer(new Layer(1, 64, 16, 16, reluLayer)); // ReLU Layer
        addLayer(new Layer(1, 64, 16, 16, 2, 2, poolingLayer)); // Pooling Layer

        addLayer(new Layer(1, 64, 8, 8, flattenLayer)); // Flatten Layer

        addLayer(new Layer(1, 1, 64*8*8, 1, 128, denseLayer)); // Dense Layer
        addLayer(new Layer(1, 1, 128, 1, batchNormLayer, "dense")); // BatchNorm Layer
        addLayer(new Layer(1, 1, 128, 1, reluLayer)); // ReLU Layer
        
        addLayer(new Layer(1, 1, 128, 1, 64, denseLayer)); // Dense Layer
        addLayer(new Layer(1, 1, 64, 1, batchNormLayer, "dense")); // BatchNorm Layer
        addLayer(new Layer(1, 1, 64, 1, reluLayer)); // ReLU Layer

        addLayer(new Layer(1, 1, 64, 1, 10, softmaxLayer)); // Softmax Layer

        output = new Tensor(1, 1, 10, 1);
        output->copy(this->model_sequence.back()->softmax->output);

        model_sequence[0]->conv->readKernelsFromFile("data/weight/conv2d_1.txt");
        model_sequence[0]->conv->readBiasFromFile("data/weight/conv2d_2.txt");
        model_sequence[4]->conv->readKernelsFromFile("data/weight/conv2d_1_1.txt");
        model_sequence[4]->conv->readBiasFromFile("data/weight/conv2d_1_2.txt");
        model_sequence[1]->batchNorm->readParamFromFile("data/weight/batch_normalization_1.txt",
                                                       "data/weight/batch_normalization_2.txt",
                                                       "data/weight/batch_normalization_3.txt",
                                                       "data/weight/batch_normalization_4.txt");
        model_sequence[5]->batchNorm->readParamFromFile("data/weight/batch_normalization_1_1.txt",
                                                       "data/weight/batch_normalization_1_2.txt",
                                                       "data/weight/batch_normalization_1_3.txt",
                                                       "data/weight/batch_normalization_1_4.txt");
        model_sequence[10]->batchNorm->readParamFromFile("data/weight/batch_normalization_2_1.txt",
                                                       "data/weight/batch_normalization_2_2.txt",
                                                       "data/weight/batch_normalization_2_3.txt",
                                                       "data/weight/batch_normalization_2_4.txt");
        model_sequence[13]->batchNorm->readParamFromFile("data/weight/batch_normalization_3_1.txt",
                                                       "data/weight/batch_normalization_3_2.txt",
                                                       "data/weight/batch_normalization_3_3.txt",
                                                       "data/weight/batch_normalization_3_4.txt");
        model_sequence[9]->dense->readWeightsFromFile("data/weight/dense_1.txt");
        model_sequence[9]->dense->readBiasFromFile("data/weight/dense_2.txt");
        model_sequence[12]->dense->readWeightsFromFile("data/weight/dense_1_1.txt");
        model_sequence[12]->dense->readBiasFromFile("data/weight/dense_1_2.txt");
        model_sequence[15]->softmax->readWeightsFromFile("data/weight/dense_2_1.txt");
        model_sequence[15]->softmax->readBiasFromFile("data/weight/dense_2_2.txt");
    }
    Tensor *forward(Tensor* input) override {
        Tensor* current_x = input;
        model_sequence[0]->conv->forward(current_x);
        current_x = model_sequence[1]->batchNorm->inference_forward(model_sequence[0]->conv->output);
        current_x = model_sequence[2]->relu->forward(current_x);
        current_x = model_sequence[3]->pooling->forward(current_x);
        current_x = model_sequence[4]->conv->forward(current_x);
        current_x = model_sequence[5]->batchNorm->inference_forward(model_sequence[4]->conv->output);
        current_x = model_sequence[6]->relu->forward(current_x);
        current_x = model_sequence[7]->pooling->forward(current_x);
        current_x = model_sequence[8]->flatten->forward(current_x);
        current_x = model_sequence[9]->dense->forward(current_x);
        current_x = model_sequence[10]->batchNorm->inference_forward(model_sequence[9]->dense->output);
        current_x = model_sequence[11]->relu->forward(current_x);
        current_x = model_sequence[12]->dense->forward(current_x);
        current_x = model_sequence[13]->batchNorm->inference_forward(model_sequence[12]->dense->output);
        current_x = model_sequence[14]->relu->forward(current_x);
        current_x = model_sequence[15]->softmax->forward(current_x);
        output = current_x;
        return current_x;
    }
    void printModelParam(){
        for(int i = 0; i < model_sequence.size(); i++){
            if(model_sequence[i]->conv){
                model_sequence[i]->conv->printParam();
            }
            else if(model_sequence[i]->dense){
                model_sequence[i]->dense->printParam();
            }
            else if(model_sequence[i]->softmax){
                model_sequence[i]->softmax->printParam();
            }
            else if(model_sequence[i]->batchNorm){
                std::cout << "BatchNorm Layer Parameters:" << std::endl;
                std::cout << "Gamma:" << std::endl;
                model_sequence[i]->batchNorm->gamma->print();
                std::cout << "Beta:" << std::endl;
                model_sequence[i]->batchNorm->beta->print();
            }
        }
    }
};
int main(){
    srand(time(0));
    CNN model;
    Tensor input(1, 3, 32, 32);
    input.readFromFile("input.txt");
    // input.print();
    Tensor* currentX = &input;
    // std::cout << "Model Parameters:" << std::endl;
    // model.printModelParam();
    currentX = model.model_sequence[0]->conv->forward(currentX);
    model.model_sequence[0]->conv->output->writeToFile("model_out/conv_output.txt");
    currentX = model.model_sequence[1]->batchNorm->inference_forward(currentX);
    model.model_sequence[1]->batchNorm->output->writeToFile("model_out/batchnorm_output.txt");
    currentX = model.model_sequence[2]->relu->forward(currentX);
    model.model_sequence[2]->relu->output->writeToFile("model_out/relu_output.txt");
    currentX = model.model_sequence[3]->pooling->forward(currentX);
    model.model_sequence[3]->pooling->output->writeToFile("model_out/pooling_output.txt");
    currentX = model.model_sequence[4]->conv->forward(currentX);
    model.model_sequence[4]->conv->output->writeToFile("model_out/conv1_output.txt");
    currentX = model.model_sequence[5]->batchNorm->inference_forward(currentX);
    model.model_sequence[5]->batchNorm->output->writeToFile("model_out/batchnorm1_output.txt");
    currentX = model.model_sequence[6]->relu->forward(currentX);
    model.model_sequence[6]->relu->output->writeToFile("model_out/relu1_output.txt");
    currentX = model.model_sequence[7]->pooling->forward(currentX);
    model.model_sequence[7]->pooling->output->writeToFile("model_out/pooling1_output.txt");
    currentX = model.model_sequence[8]->flatten->forward(currentX);
    model.model_sequence[8]->flatten->output->writeToFile("model_out/flatten_output.txt");
    currentX = model.model_sequence[9]->dense->forward(currentX);
    model.model_sequence[9]->dense->output->writeToFile("model_out/dense_output.txt");
    currentX = model.model_sequence[10]->batchNorm->inference_forward(currentX);
    model.model_sequence[10]->batchNorm->output->writeToFile("model_out/batchnorm2_output.txt");
    currentX = model.model_sequence[11]->relu->forward(currentX);
    model.model_sequence[11]->relu->output->writeToFile("model_out/relu2_output.txt");
    currentX = model.model_sequence[12]->dense->forward(currentX);
    model.model_sequence[12]->dense->output->writeToFile("model_out/dense1_output.txt");
    currentX = model.model_sequence[13]->batchNorm->inference_forward(currentX);
    model.model_sequence[13]->batchNorm->output->writeToFile("model_out/batchnorm3_output.txt");
    currentX = model.model_sequence[14]->relu->forward(currentX);
    model.model_sequence[14]->relu->output->writeToFile("model_out/relu3_output.txt");
    currentX = model.model_sequence[15]->softmax->forward(currentX);
    model.model_sequence[15]->softmax->output->writeToFile("model_out/softmax_output.txt");
    
    std::cout<<"\n\n\n\noutput";
    // model.model_sequence[7]->pooling->output->print();
    model.model_sequence[15]->softmax->output->print();
    return 0;
}