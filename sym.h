#ifndef SYM_H
#define SYM_H

#include <armadillo>
#include <libint2.hpp>

struct orbPair
{
  int orb;
  double weight;
};

void invert(libint2::Atom &a,arma::Row<int> &orb,int l,bool ispher);

void c2(int axis,arma::rowvec &a,arma::Row<int> &orb,int l,bool ispher);

void sig(int norm,arma::rowvec &a,arma::Row<int> &orb,int l,bool ispher);

void reduce(arma::rowvec r,arma::mat c2vTable,arma::rowvec &groups);

double symOp(arma::rowvec a1,arma::rowvec &a2,int l,int ml,std::string op,bool ispher);

void applySym(std::vector<libint2::Atom> atoms,std::string basis,std::string pointGroup,std::vector<libint2::Atom> &allAtoms,std::vector<arma::mat> &Cs,std::vector<std::string> &irreps,int &nMO);
void applySym2(const std::vector<libint2::Atom> &atoms,std::string basis,std::string pointGroup,std::vector<libint2::Atom> &allAtoms,std::vector<int> &nAOs,std::vector<std::vector<std::vector<orbPair> > > &ao2mo,std::vector<std::vector<std::vector<int> > > &bas,arma::Mat<int> &prodTable,std::vector<std::vector<int> > &modes);

void initPG(std::string pg,std::vector<std::string> &irreps,std::vector<std::string> &ops,arma::mat &charTable,arma::Mat<int> &prodTable);

void genAtoms(std::string basis,std::string pointGroup,const std::vector<libint2::Atom> &atoms,std::vector<libint2::Atom> &atomsFull);

void atomStartEnd(std::string pointGroup,const std::vector<libint2::Atom> &atoms,int numA,int &start,int &end);

#endif
