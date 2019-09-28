#ifndef D2_H
#define D2_H

#include <armadillo>
#include <chrono>

//Index of V2aa of D2aa, normalized to r(r-1)/2
int indexD(int i,int j,int r);
int indexR(int i,int j);
int indexD1(int i,int r);

//Index of V2ab or D2ab, normalized to r^2
int index(int i,int j,int r);

//Combinatorial, should probably use a prebuilt function
int nChooseR(int n,int r);

void setTime(std::chrono::high_resolution_clock::time_point &t1,std::chrono::high_resolution_clock::time_point &t2,std::chrono::duration<double> &time_span);

#endif
