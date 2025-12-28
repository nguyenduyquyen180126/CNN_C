#include "../include/batch_norm.h"
BatchNorm::BatchNorm(int n, int d, int h, int w, double momentum, std::string type){
    this->type = type;
    if(type == "conv"){
        this->n = n;
        this->d = d;
        this->h = h;
        this->w = w;
        this->momentum = momentum;
        this->epsilon = 0.001;
        gamma = new Matrix(d, 1, false);
        beta = new Matrix(d, 1, false);
        running_mean = new Matrix(d, 1);
        running_var = new Matrix(d, 1);
        this->output = new Tensor(n, d, h, w);
    }
    else if(type == "dense"){
        this->n = n;
        this->d = d;
        this->h = h;
        this->w = w;
        this->momentum = momentum;
        this->epsilon = 0.001;
        gamma = new Matrix(h, 1, false);
        beta = new Matrix(h, 1, false);
        running_mean = new Matrix(h, 1);
        running_var = new Matrix(h, 1);
        this->output = new Tensor(n, d, h, w);
    }
}
BatchNorm::~BatchNorm() {
    delete gamma;
    delete beta;
    delete running_mean;
    delete running_var;
}
Tensor* BatchNorm::inference_forward(Tensor* input){
    assert(input->n == this->n && input->d == this->d && input->h == this->h && input->w == this->w);
    this->input = input;
    
    if(type == "conv"){
        // For conv layers: normalize across depth dimension
        for(int batch = 0; batch < this->n; batch++){
            for(int depth = 0; depth < this->d; depth++){
                double mean = running_mean->data[depth][0];
                double var = running_var->data[depth][0];
                double gamma_val = gamma->data[depth][0];
                double beta_val = beta->data[depth][0];
                for(int i = 0; i < this->h; i++){
                    for(int j = 0; j < this->w; j++){
                        double x = input->matrix[batch][depth]->data[i][j];
                        double x_hat = (x - mean) / sqrt(var + this->epsilon);
                        double y = gamma_val * x_hat + beta_val;
                        output->matrix[batch][depth]->data[i][j] = y;
                    }
                }
            }
        }
    }
    else if(type == "dense"){
        // For dense layers: normalize across height dimension (features)
        for(int batch = 0; batch < this->n; batch++){
            for(int i = 0; i < this->h; i++){
                double mean = running_mean->data[i][0];
                double var = running_var->data[i][0];
                double gamma_val = gamma->data[i][0];
                double beta_val = beta->data[i][0];
                double x = input->matrix[batch][0]->data[i][0];
                double x_hat = (x - mean) / sqrt(var + this->epsilon);
                double y = gamma_val * x_hat + beta_val;
                output->matrix[batch][0]->data[i][0] = y;
            }
        }
    }
    return output;
}
void BatchNorm::readParamFromFile(const char* gamma_file, const char* beta_file, const char* running_mean_file, const char* running_var_file){
    FILE* file = fopen(gamma_file, "r");
    if(file == NULL){
        std::cerr << "Error opening file: " << gamma_file << std::endl;
        return;
    }
    // Read gamme size
    fscanf(file, "%d\n", &gamma->rows);
    for(int depth = 0; depth < gamma->rows; depth++){
        fscanf(file, "%lf", &gamma->data[depth][0]);
        fgetc(file);
    }
    fclose(file);

    file = fopen(beta_file, "r");
    if(file == NULL){
        std::cerr << "Error opening file: " << beta_file << std::endl;
        return;
    }
    // Read beta size
    fscanf(file, "%d\n", &beta->rows);
    for(int depth = 0; depth < beta->rows; depth++){
        fscanf(file, "%lf", &beta->data[depth][0]);
        fgetc(file);
    }
    fclose(file);

    file = fopen(running_mean_file, "r");
    if(file == NULL){
        std::cerr << "Error opening file: " << running_mean_file << std::endl;
        return;
    }
    // Read running mean size
    fscanf(file, "%d\n", &running_mean->rows);
    for(int depth = 0; depth < running_mean->rows; depth++){
        fscanf(file, "%lf", &running_mean->data[depth][0]);
        fgetc(file);
    }
    fclose(file);

    file = fopen(running_var_file, "r");
    if(file == NULL){
        std::cerr << "Error opening file: " << running_var_file << std::endl;
        return;
    }
    // Read running var size
    fscanf(file, "%d\n", &running_var->rows);
    for(int depth = 0; depth < running_var->rows; depth++){
        fscanf(file, "%lf", &running_var->data[depth][0]);
        fgetc(file);
    }
    fclose(file);
}
void BatchNorm::printParam(){
    std::cout << "Gamma:" << std::endl;
    gamma->print();
    std::cout << "Beta:" << std::endl;
    beta->print();
    std::cout << "Running Mean:" << std::endl;
    running_mean->print();
    std::cout << "Running Variance:" << std::endl;
    running_var->print();
}