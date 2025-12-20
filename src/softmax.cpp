#include "../include/softmax.h"
Softmax::Softmax(int n_input, int d_input, int h_input, int w_input, int h_output){
    this->n_input = n_input;
    this->h_input = h_input;
    this->h_output = h_output;
    output = new Tensor(n_input, 1, h_output, 1);
    weights = new Matrix(h_input, h_output, true);
    bias = new Matrix(h_output, 1, false);
    dL_dB = new Matrix(h_output, 1);
    dL_dW = new Matrix(this->weights->rows, this->weights->cols);
    dL_dX = new Matrix(this->h_input, 1);
}
Softmax::~Softmax(){
    delete output;
    delete weights;
    delete bias;
    delete dL_dB;
    delete dL_dX;
    delete dL_dW;
}
Tensor *Softmax::forward(Tensor* input){
    Matrix w_T(weights->cols, weights->rows);
    weights->transpose(&w_T);
    for(int i=0; i<input->n; i++){
        w_T.multiply(input->matrix[i][0], output->matrix[i][0]);
        output->matrix[i][0]->add(bias, output->matrix[i][0]);
        // z->matrix[i][0]->copy(output->matrix[i][0]);
        // Tim max de tranh overflow
        double maxVal = output->matrix[i][0]->data[0][0];
        for(int j = 1; j < output->matrix[i][0]->rows; ++j){
            if(output->matrix[i][0]->data[j][0] > maxVal){
                maxVal = output->matrix[i][0]->data[j][0];
            }
        }
        // Tinh exp va tong
        double sum = 0.0;
        for(int j = 0; j < output->matrix[i][0]->rows; ++j){
            output->matrix[i][0]->data[j][0] = exp(output->matrix[i][0]->data[j][0]);
            sum += output->matrix[i][0]->data[j][0];
        }
        for(int j = 0; j < output->matrix[i][0]->rows; ++j){
            output->matrix[i][0]->data[j][0] /= sum;
        }
    }
    return output;
}
void Softmax::printParam(){
    std::cout << "Softmax Layer Parameters:" << std::endl;
    std::cout << "Weights:" << std::endl;
    this->weights->print();
    std::cout << "Bias:" << std::endl;
    this->bias->print();
}
Matrix* Softmax::backward(Matrix* d_Out){
    return nullptr;
}
void Softmax::readWeightsFromFile(const char* filename){
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
void Softmax::readBiasFromFile(const char* filename){
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
    fclose(file);
}