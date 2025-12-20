#ifndef UTILS_H
#define UTILS_H
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double randn (double mu, double sigma);
void shuffle(int *reference_table, int size);
void createReferTable(int *reference_table, int size);
int isBatchComplete(int dataSetSize, int batchSize, int index);
double Re_LU(double x);
double d_ReLU(double x);
#endif
