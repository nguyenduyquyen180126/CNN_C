#include "../include/tensor.h"
Matrix::Matrix(int r, int c, bool random){
    rows = r;
    cols = c;
    data = new double*[rows];
    for(int i = 0; i < rows; ++i){
        data[i] = new double[cols];
        for(int j = 0; j < cols; ++j){
            if (random){
                data[i][j] = randn(0.0, 1.0);
            } 
            else{
                data[i][j] = 0.0;
            }
        }
    }
}
Matrix::~Matrix() {
    for(int i = 0; i < rows; ++i){
        delete[] data[i];
    }
    delete[] data;
}
void Matrix::print(){
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            printf("%8.4lf ", data[i][j]);
        }
        std::cout << std::endl;
    }
}
void Matrix::assign(double val){
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j) {
            data[i][j] = val;
        }
    }
}
void Matrix::add(Matrix* other, Matrix* result){
    assert(rows == other->rows && cols == other->cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            result->data[i][j] = data[i][j] + other->data[i][j];
        }
    }
}
void Matrix::multiply(Matrix* other, Matrix* result) {
    assert(cols == other->rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other->cols; ++j) {
            result->data[i][j] = 0.0;
            for (int k = 0; k < cols; ++k) {
                result->data[i][j] += data[i][k] * other->data[k][j];
            }
        }
    }
}
void Matrix::addNumber(double num, Matrix* result) {
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            result->data[i][j] = data[i][j] + num;
        }
    }
}
void Matrix::applyReLU(Matrix* result) {
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            result->data[i][j] = Re_LU(data[i][j]);
        }
    }
}
void Matrix::padding(Matrix* result, int pad) {
    assert(result->rows == rows + 2 * pad && result->cols == cols + 2 * pad);
    for(int i = 0; i < result->rows; ++i){
        for(int j = 0; j < result->cols; ++j){
            if(i < pad || i >= rows + pad || j < pad || j >= cols + pad){
                result->data[i][j] = 0.0;
            } 
            else{
                result->data[i][j] = data[i - pad][j - pad];
            }
        }
    }
}
void Matrix::convolve(Matrix* kernel, Matrix* result, int stride, int padding) {
    assert(result->rows == (rows - kernel->rows + 2 * padding) / stride + 1);
    assert(result->cols == (cols - kernel->cols + 2 * padding) / stride + 1);
    Matrix padded(rows + 2 * padding, cols + 2 * padding);
    this->padding(&padded, padding);
    for (int i = 0; i < result->rows; i++){
        for(int j = 0; j < result->cols; j++){
            double sum = 0.0;
            for(int m = 0; m < kernel->rows; m++){
                for(int n = 0; n < kernel->cols; n++){
                    sum += padded.data[i * stride + m][j * stride + n] * kernel->data[m][n];
                }
            }
            result->data[i][j] = sum;
        }
    }
}
void Matrix::maxPool(Matrix* result, int poolSize, int stride){
    assert(result->rows == (rows - poolSize) / stride + 1);
    assert(result->cols == (cols - poolSize) / stride + 1);
    for(int i = 0; i < result->rows; ++i){
        for(int j = 0; j < result->cols; ++j){
            double maxVal = data[i * stride][j * stride];
            for(int m = 0; m < poolSize; ++m){
                for(int n = 0; n < poolSize; ++n){
                    double currVal = data[i * stride + m][j * stride + n];
                    if(currVal > maxVal){
                        maxVal = currVal;
                    }
                }
            }
            result->data[i][j] = maxVal;
        }
    }
}
void Matrix::flatten(Matrix* result) {
    assert(result->cols == 1 && result->rows == rows * cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            result->data[i * cols + j][0] = data[i][j];
        }
    }
}
void Matrix::copy(Matrix* src) {
    assert(rows == src->rows && cols == src->cols);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            data[i][j] = src->data[i][j];
        }
    }
}
Tensor::Tensor(int n, int d, int h, int w, bool random){
    this->n = n;
    this->d = d;
    this->h = h;
    this->w = w;
    matrix = new Matrix**[n];
    for(int i = 0; i < n; ++i){
        matrix[i] = new Matrix*[d];
    }
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < d; ++j){
            matrix[i][j] = new Matrix(h, w, random);
        }
    }
}
Tensor::~Tensor(){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < d; ++j){
            delete matrix[i][j];
        }
        delete[] matrix[i];
    }
    delete[] matrix;
}
void Tensor::print(){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < d; j++){
            printf("[Batch_Num = %d][Depth = %d]:\n", i, j);
            matrix[i][j]->print();
        }
    }
}
void Tensor::convolve(Tensor* kernel, Tensor* result, int stride, int padding) {
    assert(result->d == kernel->n);
    assert(result->h == (h - kernel->h + 2 * padding) / stride + 1);
    assert(result->w == (w - kernel->w + 2 * padding) / stride + 1);
    for(int batch = 0; batch < n; ++batch){
        for(int out_depth = 0; out_depth < result->d; ++out_depth){
            result->matrix[batch][out_depth]->assign(0.0);
            for(int in_depth = 0; in_depth < d; ++in_depth){
                Matrix temp(result->h, result->w);
                matrix[batch][in_depth]->convolve(kernel->matrix[out_depth][in_depth], &temp, stride, padding);
                result->matrix[batch][out_depth]->add(&temp, result->matrix[batch][out_depth]);
            }
        }
    }
}
void Matrix::subtract(Matrix* other, Matrix* result){
    assert(this->rows == other->rows && this->cols == other->cols);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            result->data[i][j] = data[i][j] - other->data[i][j];
        }
    }
}
void Tensor::maxPool(Tensor* result, int poolSize, int stride){
    assert(result->n == n && result->d == d);
    assert(result->h == (h - poolSize) / stride + 1);
    assert(result->w == (w - poolSize) / stride + 1);
    for(int batch = 0; batch < n; ++batch){
        for(int depth = 0; depth < d; ++depth){
            matrix[batch][depth]->maxPool(result->matrix[batch][depth], poolSize, stride);
        }
    }
}
void Tensor::flatten(Tensor* result){
    assert((result->n == n && result->d == 1));
    assert(result->h == d * h * w);
    assert(result->w == 1);
    for(int batch = 0; batch < n; ++batch){
        for(int depth = 0; depth < d; ++depth){
            Matrix temp(h*w, 1);
            matrix[batch][depth]->flatten(&temp);
            for(int i = 0; i < h*w; ++i){
                result->matrix[batch][0]->data[depth * h * w + i][0] = temp.data[i][0];
            }
        }
    }
}
void Tensor::copy(Tensor* src) {
    assert(n == src->n && d == src->d && h == src->h && w == src->w);
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < d; ++j){
            matrix[i][j]->copy(src->matrix[i][j]);
        }
    }
}
void Matrix::transpose(Matrix* result) {
    assert(result->rows == cols && result->cols == rows);
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            result->data[j][i] = data[i][j];
        }
    }
}
void Tensor::assignMatrix(Matrix* mat, int batchIndex, int depthIndex) {
    assert(mat->rows == h && mat->cols == w);
    assert(batchIndex < n && depthIndex < d);
    matrix[batchIndex][depthIndex]->copy(mat);
}
void Tensor::subtract(Tensor* other, Tensor* result){
    assert(this->n == other->n);
    assert(this->d == other->d);
    assert(this->h == other->h);
    assert(this->w == other->w);
    for(int batch = 0; batch < this->n; batch++){
        for(int depth = 0; depth < this->d; depth++){
            this->matrix[batch][depth]->subtract(other->matrix[batch][depth], result->matrix[batch][depth]);
        }
    }
}
void Tensor::mergeTensorToMatrix(Matrix* result){
    assert(this->d == 1);// Ensure it has depth = 1
    for(int batch = 0; batch < this->n; batch++){
        result->add(this->matrix[batch][0], result);
    }
}
void Tensor::readFromFile(const char * filename){
    FILE* file = fopen(filename, "r");
    if(file == NULL){
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    // Read tensor dimensions from first line of CSV file
    // Format in file is: n,h,w,d
    int file_n, file_h, file_w, file_d;
    fscanf(file, "%d,%d,%d,%d\n", &file_n, &file_h, &file_w, &file_d);
    
    // Our tensor format is (n, d, h, w)
    assert(file_n == n && file_d == d && file_h == h && file_w == w);
    
    for(int batch = 0; batch < n; batch++){
        for(int depth = 0; depth < d; depth++){
            for(int i = 0; i < h; i++){
                for(int j = 0; j < w; j++){
                    fscanf(file, "%lf", &matrix[batch][depth]->data[i][j]);
                    if(!(batch == n-1 && depth == d-1 && i == h-1 && j == w-1)){
                        fgetc(file); // consume comma or newline
                    }
                }
            }
        }
    }
    fclose(file);
}
void Tensor::writeToFile(const char * filename){
    FILE* file = fopen(filename, "w");
    if(file == NULL){
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    // In voi formart: n,h,w,d cho trung dinh dang cua tf
    fprintf(file, "%d,%d,%d,%d\n", n, h, w, d);
    
    for(int batch = 0; batch < n; batch++){
        for(int depth = 0; depth < d; depth++){
            for(int i = 0; i < h; i++){
                for(int j = 0; j < w; j++){
                    fprintf(file, "%lf", matrix[batch][depth]->data[i][j]);
                    if(!(batch == n-1 && depth == d-1 && i == h-1 && j == w-1)){
                        fprintf(file, ",");
                    }
                }
                fprintf(file, "\n");
            }
        }
    }
    fclose(file);
}