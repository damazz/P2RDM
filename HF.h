#ifndef HF_H
#define HF_H

#include <armadillo>
#include "D2.h"

void  buildD1sym(const std::vector<int> ns,const std::vector<arma::mat> &Cs,std::vector<arma::mat> &D1s);

void buildGsym(std::vector<arma::mat> &Gs,const std::vector<arma::mat> &D1s,std::vector<std::vector<arma::mat> > &W);

void rotateInts(std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V2s,const std::vector<arma::mat> &Cs,std::vector<std::vector<std::vector<int> > > bas,std::vector<int> nOcc,std::vector<int> nCore,double &Ecore);

void DIIS(std::vector<arma::mat> &Fn,const std::vector<arma::mat> &D1,const std::vector<arma::mat> &S,std::vector<std::vector<arma::mat> > &Fs,std::vector<std::vector<arma::mat> > &Es,arma::mat &B,int nF);

double solveHFsym(int nels,std::vector<arma::mat> &Cs,std::vector<arma::mat> &Ss,std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V2f,const std::vector<std::vector<std::vector<int> > > &bas,std::vector<int> &ns,std::vector<int> &nCore,std::vector<int> nAOs,bool print,int core,bool useOld);

void minTheta(const arma::mat &S,const arma::mat &Cold,const arma::vec &eigs,arma::mat &Cnew,bool flag,int nels);
#endif
