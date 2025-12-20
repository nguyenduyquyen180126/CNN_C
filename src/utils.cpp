#include "../include/utils.h"
double randn (double mu, double sigma){
    static double X2;
    static int call = 0;
    double U1, U2, W, mult;

    if (call) {
        call = 0;
        return mu + sigma * X2;
    }

    do {
        U1 = -1.0 + 2.0 * ((double) rand() / RAND_MAX);
        U2 = -1.0 + 2.0 * ((double) rand() / RAND_MAX);
        W = U1 * U1 + U2 * U2;
    } while (W >= 1.0 || W == 0.0);

    mult = sqrt((-2.0 * log(W)) / W);
    X2 = U2 * mult;
    call = 1;
    return mu + sigma * (U1 * mult);
}
void shuffle(int *reference_table, int size){
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1); // chọn ngẫu nhiên j trong [0, i]
        // hoán đổi arr[i] và arr[j]
        int temp = reference_table[i];
        reference_table[i] = reference_table[j];
        reference_table[j] = temp;
    }
}
void createReferTable(int *reference_table, int size){
    for(int i=0; i<size; i++){
        reference_table[i] = i;
    }
}
int isBatchComplete(int dataSetSize, int batchSize, int index){
    if (batchSize <= 0) {
        return 0;
    }
    if (index == dataSetSize - 1) {
        return 1;
    }
    for (int i = 0; i < dataSetSize / batchSize; i++) {
        if (i * batchSize == index) {
            return 1;
        }
    }
    return 0;
}
double Re_LU(double x){
    if(x >= 0)return x;
    else{
        return 0.01*x;
    }
}
double d_ReLU(double x){
    if(x >= 0){
        return 1;
    }
    else{
        return 0.01;
    }
}