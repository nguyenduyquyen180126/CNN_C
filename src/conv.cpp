#include "../include/conv.h"
Conv::Conv(int n_input, int d_input, int h_input, int w_input, int n_kernels, int kernel_size, int stride, int padding){
    this->n_input = n_input;
    this->d_input = d_input;
    this->h_input = h_input;
    this->w_input = w_input;
    this->n_kernels = n_kernels;
    this->kernel_size = kernel_size;
    this->stride = stride;
    this->padding = padding;
    this->n_output = n_input;
    this->d_output = n_kernels;
    this->h_output = (h_input - kernel_size + 2 * padding) / stride + 1;
    this->w_output = (w_input - kernel_size + 2 * padding) / stride + 1;
    input = new Tensor(n_input, d_input, h_input, w_input);
    output = new Tensor(n_output, d_output, h_output, w_output);
    kernels = new Tensor(n_kernels, d_input, kernel_size, kernel_size, true);
    bias = new Matrix(n_kernels, 1, false);
}
Conv::~Conv(){
    delete input;
    delete output;
    delete kernels;
    delete bias;
}
Tensor* Conv::forward(Tensor* input){
    assert(input->h == h_input && input->w == w_input);
    this->input->copy(input);
    this->input->convolve(kernels, output, stride, padding);
    for(int n = 0; n < output->n; ++n){
        for(int d = 0; d < output->d; ++d){
            output->matrix[n][d]->addNumber(bias->data[d][0], output->matrix[n][d]);
        }
    }
    return output;
}
void Conv::printParam(){
    std::cout << "Convolutional Layer Parameters:" << std::endl;
    std::cout << "Number of Kernels: " << n_kernels << std::endl;
    std::cout << "Kernel Size:" << kernel_size << std::endl;
    this->kernels->print();
    std::cout << "Bias:" << std::endl;
    this->bias->print();
    std::cout << "Stride:" << stride << std::endl;
    std::cout << "Padding:" << padding << std::endl;
    std::cout << "Input Dimensions: (" << n_input << ", " << d_input << ", " << h_input << ", " << w_input << ")" << std::endl;
    std::cout << "Output Dimensions: (" << n_output << ", " << d_output << ", " << h_output << ", " << w_output << ")" << std::endl;
}
void Conv::readKernelsFromFile(const char* filename){
    FILE* file = fopen(filename, "r");
    if(file == NULL){
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    // Read the number of kernels, depth, height, width
    int n, d, h, w;
    fscanf(file, "%d,%d,%d,%d\n", &h, &w, &d, &n);
    for(int num_kernel = 0; num_kernel < n; ++num_kernel){
        for(int depth = 0; depth < d; ++depth){
            for(int i = 0; i < h; ++i){
                for(int j = 0; j < w; ++j){
                    if(fscanf(file, "%lf", &this->kernels->matrix[num_kernel][depth]->data[i][j]) != 1){
                        std::cerr << "Error reading kernel value at kernel "<< num_kernel << ", depth " << depth << ", position (" << i << ", " << j << ")" << std::endl;
                        fclose(file);
                        return;
                    }
                    // Skip comma if not the last element
                    fgetc(file); // lay , va xuong dong
                }
            }
        }
    }
}
void Conv::readBiasFromFile(const char* filename){
    FILE* file = fopen(filename, "r");
    if(file == NULL){
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    int n;
    fscanf(file, "%d\n", &n);
    
    if(n != this->n_kernels){
        std::cerr << "Warning: bias file has " << n << " values but expected " << this->n_kernels << std::endl;
    }
    
    // Read all bias values
    for(int i = 0; i < this->n_kernels && i < n; ++i){
        if(fscanf(file, "%lf", &this->bias->data[i][0]) != 1){
            std::cerr << "Error reading bias value at index " << i << std::endl;
            fclose(file);
            return;
        }
        // Skip comma if not the last element
        if(i < n - 1){
            fgetc(file); // consume comma
        }
    }
    
    fclose(file);
}