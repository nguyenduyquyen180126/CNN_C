#include "../include/dense.h"
Dense::Dense(int n_input, int d_input, int h_input, int w_input, int h_output){
    this->n_input = n_input;
    this->d_input = d_input;
    this->h_input = h_input;
    this->w_input = w_input;
    this->n_output = n_input;
    this->d_output = d_input;
    this->h_output = h_output;
    this->w_output = 1;
    weights = new Matrix(h_input, h_output, true);
    bias = new Matrix(h_output, 1, false);
    input = new Tensor(n_input, d_input, h_input, w_input);
    output = new Tensor(n_output, d_output, h_output, w_output);
}
Dense::~Dense(){
    delete weights;
    delete bias;
    delete input;
    delete output;
}
Tensor *Dense::forward(Tensor* input){
    assert(input->h == h_input);
    this->input->copy(input);
    for(int batch = 0; batch < n_input; ++batch){
        Matrix w_T(h_output, h_input);
        weights->transpose(&w_T);
        w_T.multiply(input->matrix[batch][0], output->matrix[batch][0]);
        output->matrix[batch][0]->add(bias, output->matrix[batch][0]);
    }
    return output;
}
Matrix* Dense::backward(Matrix* d_Out, double lr){
    return nullptr;
}
void Dense::printParam(){
    std::cout << "Dense Layer Parameters:" << std::endl;
    std::cout << "Weights:" << std::endl;
    this->weights->print();
    std::cout << "Bias:" << std::endl;
    this->bias->print();
}
void Dense::readWeightsFromFile(const char* filename){
    FILE* file = fopen(filename, "r");
    if(file == NULL){
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    // Read weights dimensions
    int rows, cols;
    fscanf(file, "%d,%d\n", &rows, &cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            fscanf(file, "%lf", &weights->data[i][j]);
            fgetc(file); // consume comma or newline
        }
    }
    fclose(file);
}
void Dense::readBiasFromFile(const char* filename){
    FILE* file = fopen(filename, "r");
    if(file == NULL){
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    // Read bias dimensions
    int rows;
    fscanf(file, "%d\n", &rows);
    for(int i = 0; i < rows; ++i){
        fscanf(file, "%lf", &bias->data[i][0]);
        fgetc(file); // consume comma or newline
    }
}