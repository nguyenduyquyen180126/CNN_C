// #include "../include/dataloader.h"
// void loadData(const char* filePath, std::vector<Tensor> *X, std::vector<Tensor> *Y){
//     FILE *fp = fopen(filePath, "r");
//     assert(fp != NULL && "File không tồn tại");
//     int **data = (int **)malloc(MAX_ROWS * sizeof(int *));
//     if (!data) {
//         perror("Không thể cấp phát hàng");
//         exit(EXIT_FAILURE);
//     }

//     char line[MAX_LINE_LENGTH];
//     int row = 0;
//     int col;
//     while (fgets(line, sizeof(line), fp) && row < MAX_ROWS) {
//         data[row] = (int *)malloc(MAX_COLS * sizeof(int));
//         if (!data[row]) {
//             perror("Không thể cấp phát cột");
//             exit(EXIT_FAILURE);
//         }

//         // line[strcspn(line, "\r\n")] = 0;

//         char *token = strtok(line, ",");
//         col = 0;
//         while (token != NULL && col < MAX_COLS) {
//             data[row][col] = atoi(token);
//             token = strtok(NULL, ",");
//             col++;
//         }
//         row++;
//     }
//     for(int i=0; i<row; i++){
//         int label = data[i][0];
//         Tensor y(10, 1);
//         for(int j=0; j<y.size; j++){
//             if(j == label) y.dat[j] = 1;
//             else y.dat[j] = 0;
//         }
//         Y->push_back(y);
//         Tensor x(784, 1);
//         for(int j=1; j<col; j++){
//             x.dat[j] = data[i][j]/255.0;
//         }
//         X->push_back(x);
//     }
// }
// void loadData2D(const char* filePath, std::vector<Tensor> *X, std::vector<Tensor> *Y){
    
// }
// void splitData(std::vector<Tensor> *X, 
//                 std::vector<Tensor> *Y, 
//                 std::vector<Tensor> *X_train, 
//                 std::vector<Tensor> *X_val,
//                 std::vector<Tensor> *X_test,
//                 std::vector<Tensor> *Y_train,
//                 std::vector<Tensor> *Y_val,
//                 std::vector<Tensor> *Y_test
//                 )
// {
//     int trainSize = 0.7*X->size();
//     int valSize = 0.2*X->size();
//     int testSize = X->size() - (trainSize + valSize);
//     for(int i=0; i<trainSize; i++){
//         X_train->push_back((*X)[i]);
//         Y_train->push_back((*Y)[i]);
//     }
//     for(int i=trainSize; i<trainSize + valSize; i++){
//         X_val->push_back((*X)[i]);
//         Y_val->push_back((*Y)[i]);
//     }
//     for(int i=trainSize + valSize; i<trainSize + valSize + testSize; i++){
//         X_test->push_back((*X)[i]);
//         Y_test->push_back((*Y)[i]);
//     }
// }