#include "stdafx.h"
#include <armadillo>
#include <iomanip>
#include "HF.h"
#include "D2.h"
#include "math.h"
#include "time.h"
#include "sym.h"
#include "optimization.h"
#include <ctime>
#include <ratio>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// Libint Gaussian integrals library
#include <libint2.hpp>

using namespace alglib;

void readIntsGAMESS(std::string name,int nAO,arma::mat &S,arma::mat &K1);
void readInts2GAMESS(std::string name,int nAO,arma::mat &V2ab,arma::mat &V2aa);

void genInts(double &Enuc,const std::vector<libint2::Atom> &atoms,std::string basis,std::vector<arma::mat> &Ss,std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V2s,const std::vector<std::vector<std::vector<orbPair> > > &ao2mo,const std::vector<std::vector<std::vector<int> > > &bas,std::vector<int> nAOs,int &nels,arma::Mat<int> prodTable);
void printIts(const real_1d_array &x,double func,void *ptr);
void printItsNuc(const real_1d_array &x,double func,void *ptr);
void evalE(const real_1d_array &T,double &E,real_1d_array &grad,void *ptr);
void runPara(const std::vector<libint2::Atom> &atoms,std::string basis,const std::vector<int> &nAOs,const std::vector<std::vector<std::vector<orbPair> > > &ao2mo,const std::vector<std::vector<std::vector<int> > > &bas,const arma::Mat<int> &prodTable,double &E,arma::mat &T1,arma::mat &T2aa,arma::mat &T2ab,arma::mat &K1,arma::mat &V2aa,arma::mat &V2ab,bool needT,bool print);

struct intBundle
{
  double E;
  double Enuc;
  std::vector<arma::mat> K1s;
  std::vector<std::vector<arma::mat> > V2s;
  std::vector<int> nOcc;
  bool print;
  int nels;
  int virt;
  int numIt;
  arma::mat K1i_j,K1a_b,K1ia;
  arma::mat V2abi_jka,V2abia_jb,V2abh_fgn;
  arma::mat V2aai_jka,V2aaia_jb,V2aah_fgn;
  arma::mat V2aaij_kl,V2aaab_cd;
  arma::mat V2abij_kl,V2abab_cd;
  arma::mat V2abia_jb2;
  arma::mat V2abij_ab,V2aaij_ab;
  std::vector<std::chrono::duration<double> > ts;
  std::vector<std::string> tLabel;
  double Ehf;
  double maxT;
  double oldE;
};

void setMats(intBundle &i);

struct bundleNuc
{
  std::vector<libint2::Atom> atoms;
  std::string basis;
  std::string pointGroup;
  std::vector<int> nAOs;
  std::vector<std::vector<std::vector<orbPair> > > ao2mo;
  std::vector<std::vector<std::vector<int> > > bas;
  arma::Mat<int> prodTable;
  int numIt;
  double E;
};

void numEval(const real_1d_array &R,double &E,void *ptr);
void gradEval(const real_1d_array &R,double &E,real_1d_array &grad,void *ptr);

struct intBundle2
{
  double E;
  double Enuc;
  arma::mat K1;
  arma::mat V2aa;
  arma::mat V2ab;
  int nels;
  int virt;
  int numIt;
  bool print;
  time_t timE,timG,timF;
  arma::mat K1i_j,K1a_b,K1ia;
  arma::mat V1i_j,V1ia,V1a_b,V1i_a;
  arma::mat V2abi_jka,V2abia_jb,V2abh_fgn;
  arma::mat V2aai_jka,V2aaia_jb,V2aah_fgn;
  arma::mat V1aia,V1ai_j,V1aa_b;
  arma::mat V2aaij_kl,V2aaab_cd;
  arma::mat V2abij_kl,V2abab_cd;
  arma::mat V2abia_jb2;
  arma::mat V2abij_ab,V2aaij_ab;
  std::chrono::duration<double> ts1,ts2,ts3,ts4,ts5,ts6,ts7,ts8,ts9,ts10;
  std::chrono::duration<double> ts11,ts12,ts13,ts14,ts15,ts16,ts17,ts18,ts19,ts20;
  std::chrono::duration<double> ts21,ts22,ts23,ts24,ts25,ts26,ts27,ts28,ts29;
  std::vector<std::chrono::duration<double> > ts;
  double Ehf;
};

void setMats2(intBundle2 &i);
void makeD2(const arma::mat &T1,const arma::mat &T2aa,const arma::mat &T2ab,arma::mat &D1,arma::mat &D2aa,arma::mat &D2ab,intBundle2 &integrals);

int main(int argc, char *argv[])
{
  using arma::mat;
  using arma::vec;
  using namespace std::chrono;

  double x;
  int nels,nCore,nstate;
  double Ecore,Enuc;

  std::cout << std::setprecision(15);

  std::ifstream input_file(argv[1]);
  std::vector<libint2::Atom> atomsUnique = libint2::read_dotxyz(input_file);
  std::vector<libint2::Atom> atoms;

  std::string basis=std::string(argv[2]);  

  std::string pointGroup;
  std::vector<std::vector<std::vector<orbPair> > > ao2mo;
  std::vector<std::vector<std::vector<int> > > bas;
  arma::Mat<int> prodTable;
  std::vector<int> nAOs;
  if(argc > 3) //apply symmetry
  {
    pointGroup = std::string(argv[3]);
    applySym2(atomsUnique,basis,pointGroup,atoms,nAOs,ao2mo,bas,prodTable);
  }
  else //C1
  {
    pointGroup = "c1";
    applySym2(atomsUnique,basis,pointGroup,atoms,nAOs,ao2mo,bas,prodTable);
  }

//  arma::mat T1,T2aa,T2ab;
//  runPara(atoms,basis,pointGroup,nAOs,ao2mo,bas,prodTable,T1,T2aa,T2ab,true);

  bundleNuc integrals={atomsUnique,basis,pointGroup,nAOs,ao2mo,bas,prodTable,0,0.0};

  double* Ts;
  int numTot=(atomsUnique.size()-1)*3;
  Ts = new double[numTot];
  for(int n=1;n<atomsUnique.size();n++)
  {
    Ts[(n-1)*3]=atomsUnique[n].x;
    Ts[(n-1)*3+1]=atomsUnique[n].y;
    Ts[(n-1)*3+2]=atomsUnique[n].z;
  }

  real_1d_array T;
  T.setcontent(numTot,Ts);
  real_1d_array G=T;

  double E;
  gradEval(T,E,G,&integrals);
/*  double epsg = 0.00005;
  double epsf = 0;
  double epsx = 0;
  ae_int_t maxits = 0;
  minlbfgsstate state;
  minlbfgsreport rep;

  minlbfgscreatef(3,T,0.000001,state);
  minlbfgssetcond(state,epsg,epsf,epsx,maxits);
  minlbfgssetxrep(state,true);

  
//  std::cout << "Parametric 2-RDM calculation" << std::endl;
//  std::cout << "Iter\tE Corr\t\tdE\t\t\tlargest dE/dT" << std::endl;
  minlbfgsoptimize(state,numEval,printItsNuc,&integrals);
*/  //minlbfgsresults(state,T,rep);

  return EXIT_SUCCESS;
}

void printIts(const real_1d_array &x,double E,void *ptr)
{
  intBundle *iPtr=static_cast<intBundle*>(ptr);
  if((*iPtr).print)
  {
    printf("%d\t%.9f\t%.10E\t%.10E\n",(*iPtr).numIt+1,(*iPtr).E,(*iPtr).E-(*iPtr).oldE,(*iPtr).maxT);
  }
  (*iPtr).numIt++;
  (*iPtr).oldE=(*iPtr).E;

  return;
}

void printItsNuc(const real_1d_array &x,double E,void *ptr)
{
  bundleNuc *bPtr=static_cast<bundleNuc*>(ptr);
  std::cout << "Iteration:\t" << (*bPtr).numIt+1 << std::endl;
  std::cout << "Energy:\t" << (*bPtr).E << std::endl;
  std::cout << "Unique atoms (bohr):" << std::endl;
  std::cout << "Z\tx\ty\tz\n";
  for(int n=0;n<(*bPtr).atoms.size();n++)
  {
    std::cout << (*bPtr).atoms[n].atomic_number << '\t';
    std::cout << (*bPtr).atoms[n].x << '\t';
    std::cout << (*bPtr).atoms[n].y << '\t';
    std::cout << (*bPtr).atoms[n].z << std::endl;
  }

  (*bPtr).numIt++;

  return;
}

void evalE(const real_1d_array &T,double &E,real_1d_array &grad,void *ptr)
{
  using namespace std::chrono;
  std::chrono::high_resolution_clock::time_point t1,t2,t3;
  std::chrono::duration<double> time_span;

  setTime(t1,t1,time_span);
  setTime(t3,t3,time_span);

  intBundle *iPtr=static_cast<intBundle*>(ptr);
  int nels = (*iPtr).nels;
  int virt = (*iPtr).virt;
  int nact = nels+virt;
  int n1a=nels*virt;

  arma::mat T1(nels,virt);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      T1(m,e)=T[e*nels+m];
    }
  }
  arma::mat T2aa(nels*(nels-1)/2,virt*(virt-1)/2);
  int n2aa=T2aa.n_rows*T2aa.n_cols;
  for(int a=0;a<virt;a++)
  {
    for(int b=a+1;b<virt;b++)
    {
      for(int i=0;i<nels;i++)
      {
	for(int j=i+1;j<nels;j++)
	{
	  T2aa(indexD(i,j,nels),indexD(a,b,virt))=T[n1a+indexD(a,b,virt)*T2aa.n_rows+indexD(i,j,nels)];
	}
      }
    }
  }
  arma::mat T2ab(nels*nels,virt*virt);
  for(int a=0;a<virt;a++)
  {
    for(int b=0;b<virt;b++)
    {
      for(int i=0;i<nels;i++)
      {
	for(int j=0;j<nels;j++)
	{
	  T2ab(i*nels+j,a*virt+b)=T[n1a+n2aa+(a*virt+b)*T2ab.n_rows+(i*nels+j)];
	}
      }
    }
  }

  for(int i=0;i<nels*virt+nels*(nels-1)/2*virt*(virt-1)/2+nels*nels*virt*virt;i++)
  {
    grad[i]=0.0;
  }


  double x,y;
  int n1,n2,n3,n4,n5,n6,n7,n8;
  arma::mat U,Y;

  (*iPtr).E=0.0;
  int nT=0;

  arma::mat F1m(nels,virt);
  arma::mat F2aaM(nels*(nels-1)/2,virt*(virt-1)/2);
  arma::mat F2abM(nels*nels,virt*virt);
  for(int b=0;b<virt;b++)
  {
    for(int j=0;j<nels;j++)
    {
      F1m(j,b)=1.0+T1(j,b)*T1(j,b);
    }
  }
  n1=0;
  for(int a=0;a<virt;a++)
  {
    for(int b=0;b<virt;b++)
    {
      n1=a*virt+b;
      n2=0;
      for(int i=0;i<nels;i++)
      {
	for(int j=0;j<nels;j++)
	{
	  F2abM(n2,n1) = 1.0-3.0*T2ab(n2,n1)*T2ab(n2,n1);
	  F2abM(n2,n1) += T1(i,a)*T1(i,a) + T1(j,b)*T1(j,b);
	  n2++;
	}
      }
      n1++;
    }
  }
  n1=0;
  for(int a=0;a<virt;a++)
  {
    for(int b=a+1;b<virt;b++)
    {
      n2=0;
      for(int i=0;i<nels;i++)
      {
	for(int j=i+1;j<nels;j++)
	{
	  F2aaM(n2,n1) = 1.0-3.0*T2aa(n2,n1)*T2aa(n2,n1);
	  F2aaM(n2,n1) += T1(i,a)*T1(i,a) + T1(i,b)*T1(i,b) + T1(j,a)*T1(j,a) + T1(j,b)*T1(j,b);
	  n2++;
	}
      }
      n1++;
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "Initial stuff";
  nT++; t1=t2;

  for(int m=0;m<nels;m++)
  {
    x = -dot(T1.row(m),T1.row(m));
    //x = W1_i_j(m,m);
    for(int a=0;a<virt;a++)
    {
      F1m(m,a) += x;
      for(int n=0;n<nels;n++)
      {
	n3=m*nels+n;
	n5=n*nels+m;
	for(int b=0;b<virt;b++)
	{
	  n4=a*virt+b;
	  F2abM(n3,n4) += x;
	  F2abM(n5,n4) += x;
	  //F2abM(m*nels+n,a*virt+b) += x;
	  //F2abM(n*nels+m,a*virt+b) += x;
	}
      }
      for(int b=a+1;b<virt;b++)
      {
	n2 = indexD(a,b,virt);
	for(int n=0;n<m;n++)
	{
	  n1 = indexD(n,m,nels);
	  F2aaM(n1,n2) += x;
	  //F2aaM(indexD(n,m,nels),indexD(a,b,virt)) += x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  n1= indexD(m,n,nels);
	  F2aaM(n1,n2) += x;
	  //F2aaM(indexD(m,n,nels),indexD(a,b,virt)) += x;
	}
      }
    }
  }
  for(int m=0;m<nels;m++)
  {
    for(int n=0;n<nels;n++)
    {
      n1 = m*nels+n;
      x = dot(T2ab.row(n1),T2ab.row(n1));
      for(int e=0;e<virt;e++)
      {
	for(int f=0;f<virt;f++)
	{
	  n2 = e*virt+f;
	  F2abM(n1,n2) -= x;
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "Initial F";
  nT++; t1=t2;

  (*iPtr).E += trace((*iPtr).V2aaij_kl*T2aa*T2aa.t());
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      n1 = indexD(m,n,nels);
      x = dot(T2aa.row(n1),T2aa.row(n1));
      //x = Waa_ij_kl(n1,n1);
      //x = W(indexD(m,n,nels),indexD(m,n,nels));
      for(int e=0;e<virt;e++)
      {
	for(int f=e+1;f<virt;f++)
	{
	  n2=indexD(e,f,virt);
	  F2aaM(n1,n2) -= x;
	  //F2aaM(indexD(m,n,nels),indexD(e,f,virt)) -= x;
	}
      }
    }
  }
  for(int a=0;a<virt;a++)
  {
    x = dot(T1.col(a),T1.col(a));
    //x = W1_a_b(a,a);
    for(int m=0;m<nels;m++)
    {
      F1m(m,a) -= x;
      for(int n=0;n<nels;n++)
      {
	n1=m*nels+n;
	for(int b=0;b<virt;b++)
	{
	  n2=a*virt+b;
	  n3=b*virt+a;
	  F2abM(n1,n2) -= x;
	  F2abM(n1,n3) -= x;
	  //F2abM(m*nels+n,a*virt+b) -= x;
	  //F2abM(m*nels+n,b*virt+a) -= x;
	}
      }
      for(int n=m+1;n<nels;n++)
      {
	n1 = indexD(m,n,nels);
	for(int b=0;b<a;b++)
	{
	  n2=indexD(b,a,virt);
	  F2aaM(n1,n2) -= x;
	  //F2aaM(indexD(m,n,nels),indexD(b,a,virt)) -= x;
	}
	for(int b=a+1;b<virt;b++)
	{
	  n2=indexD(a,b,virt);
	  F2aaM(n1,n2) -= x;
	  //F2aaM(indexD(m,n,nels),indexD(a,b,virt)) -= x;
	}
      }
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {	
      n1=(e+nels)*nact+f+nels;
      n2=e*virt+f;
      x = dot(T2ab.col(n2),T2ab.col(n2));
      for(int m=0;m<nels;m++)
      {
	for(int n=0;n<nels;n++)
	{
	  n7=m*nels+n;
	  F2abM(n7,n2) -= x;
	  //F2abM(m*nels+n,e*virt+f) -= x;
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "More Fs";
  nT++; t1=t2;


  (*iPtr).E += trace((*iPtr).V2aaab_cd*T2aa.t()*T2aa);

  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1 = indexD(e,f,virt);
      x = dot(T2aa.col(n1),T2aa.col(n1));
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n1) -= x;
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "Still F, some E";
  nT++; t1=t2;

  arma::mat Wab_ia_jb2(nels*virt,nels*virt);
  for(int a=0;a<virt;a++)
  {
    for(int b=0;b<virt;b++)
    {
      n3=a*virt+b;
      for(int i=0;i<nels;i++)
      {
	n1=a*nels+i;
	for(int j=0;j<nels;j++)
	{
	  Wab_ia_jb2(b*nels+j,n1) = T2ab(j*nels+i,n3);
	  //W(j*virt+b,i*virt+a) = T2ab(j*nels+i,a*virt+b);
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "make Wab_ia_jb2";
  nT++; t1=t2;


  (*iPtr).E -= 2.0*trace((*iPtr).V2abia_jb2*Wab_ia_jb2*Wab_ia_jb2.t());

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "use Wab_ia_jb2";
  nT++; t1=t2;

  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n1=e*nels+m;
      x = dot(Wab_ia_jb2.row(n1),Wab_ia_jb2.row(n1));
      for(int f=0;f<virt;f++)
      {
	n2=e*virt+f;
	n3=f*virt+e;
	for(int n=0;n<nels;n++)
	{
	  F2abM(m*nels+n,n3) -= x;
	  F2abM(n*nels+m,n2) -= x;
	  //F2abM(m*nels+n,f*virt+e) -= x;
	  //F2abM(n*nels+m,e*virt+f) -= x;
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "F again";
  nT++; t1=t2;


  U=(*iPtr).V2abia_jb2*Wab_ia_jb2;
  Wab_ia_jb2.reset();
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels;n++)
	{
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+(m*nels+n)] -= 2.0*(U(f*nels+m,e*nels+n)+U(e*nels+n,f*nels+m)); 
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "VWabia_jb2";
  nT++; t1=t2;

  arma::mat Wab_ia_jb(nels*virt,nels*virt);
  for(int a=0;a<virt;a++)
  {
    for(int i=0;i<nels;i++)
    {
      n1=a*nels+i;
      for(int b=0;b<virt;b++)
      {
	n2=a*virt+b;
	for(int j=0;j<nels;j++)
	{
	  Wab_ia_jb(b*nels+j,n1) = T2ab(i*nels+j,n2);
	  //W(j*virt+b,i*virt+a) = T2ab(i*nels+j,a*virt+b);
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "make Wab_ia_jb";
  nT++; t1=t2;

  (*iPtr).E -= 2.0*trace((*iPtr).V2aaia_jb*Wab_ia_jb*Wab_ia_jb.t());

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "VWWia_jb";
  nT++; t1=t2;

  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n1=e*nels+m;
      x=dot(Wab_ia_jb.col(n1),Wab_ia_jb.col(n1));
      F1m(m,e) += x;
      //x = U(m*virt+e,m*virt+e);
      for(int f=0;f<virt;f++)
      {
	n2=e*virt+f;
	n3=f*virt+e;
	for(int n=0;n<nels;n++)
	{
	  F2abM(n*nels+m,n3) -= x;
	  F2abM(m*nels+n,n2) -= x;
	  //F2abM(n*nels+m,f*virt+e) -= x;
	  //F2abM(m*nels+n,e*virt+f) -= x;
	}
      }
      for(int f=0;f<e;f++)
      {
	n2=indexD(f,e,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n2) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n2) -= x;
	}
      }
      for(int f=e+1;f<virt;f++)
      {
	n2=indexD(e,f,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n2) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n2) -= x;
	}
      }
    }
  }
  
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "F F F F";
  nT++; t1=t2;

  arma::mat Waa_ia_jb(nels*virt,nels*virt);
  for(int a=0;a<virt;a++)
  {
    for(int i=0;i<nels;i++)
    {
      n3=a*nels+i;
      for(int m=0;m<nels;m++)
      {
	Waa_ia_jb(n3,a*nels+m)=0.0;
	//V(i*virt+a,i*virt+b)=0.0;
      }
      for(int f=0;f<a;f++)
      {
	n1=indexD(f,a,virt);
	Waa_ia_jb(n3,f*nels+i)=0.0;
	for(int m=0;m<i;m++)
	{
	  Waa_ia_jb(n3,f*nels+m) = T2aa(indexD(m,i,nels),n1);
	  //V(i*virt+a,m*virt+f) = T2aa(indexD(m,i,nels),indexD(f,a,virt));
	}
	for(int m=i+1;m<nels;m++)
	{
	  Waa_ia_jb(n3,f*nels+m) = -T2aa(indexD(i,m,nels),n1);
	}
      }
      for(int f=a+1;f<virt;f++)
      {
	n1=indexD(a,f,virt);
	Waa_ia_jb(n3,f*nels+i)=0.0;
	for(int m=0;m<i;m++)
	{
	  Waa_ia_jb(n3,f*nels+m) = -T2aa(indexD(m,i,nels),n1);
	}
	for(int m=i+1;m<nels;m++)
	{
	  Waa_ia_jb(n3,f*nels+m) = T2aa(indexD(i,m,nels),n1);
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "make Waa_ia_jb";
  nT++; t1=t2;

  (*iPtr).E -= 2.0*trace((*iPtr).V2aaia_jb*Waa_ia_jb.t()*Waa_ia_jb);

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "VWWaa_ia_jb";
  nT++; t1=t2;

  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n3=e*nels+m;
      //x=Vaa_ia_jb(n3,n3);
      x = dot(Waa_ia_jb.col(n3),Waa_ia_jb.col(n3));
      F1m(m,e) += x;
      //x = U(m*virt+e,m*virt+e);
      for(int f=0;f<virt;f++)
      {
	n4=e*virt+f;
	n5=f*virt+e;
	for(int n=0;n<nels;n++)
	{
	  F2abM(m*nels+n,n4) -= x;
	  F2abM(n*nels+m,n5) -= x;
	  //F2abM(m*nels+n,e*virt+f) -= x;
	  //F2abM(n*nels+m,f*virt+e) -= x;
	}
      }
      for(int f=0;f<e;f++)
      {
	n1 = indexD(f,e,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n1) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n1) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
      }
      for(int f=e+1;f<virt;f++)
      {
	n1 = indexD(e,f,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n1) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n1) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
      }
    }
  }

  arma::mat W1ia_jb(nels*virt,nels*virt);
  for(int b=0;b<virt;b++)
  {
    for(int j=0;j<nels;j++)
    {
      for(int a=0;a<virt;a++)
      {
	for(int i=0;i<nels;i++)
	{
	  W1ia_jb(b*nels+j,a*nels+i) = T1(i,a)*T1(j,b);
	}
      }
    }
  }

  (*iPtr).E += 4.0*trace((*iPtr).V2abia_jb*Wab_ia_jb*Waa_ia_jb);
  (*iPtr).E += 2.0*trace((*iPtr).V2abia_jb*W1ia_jb);
  (*iPtr).E -= 2.0*trace((*iPtr).V2aaia_jb*W1ia_jb);

  W1ia_jb.reset();

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "F,W1,E";
  nT++; t1=t2;

  U=(*iPtr).K1ia*(Wab_ia_jb+Waa_ia_jb);
  //U=((*iPtr).K1ia+(*iPtr).V1ia+(*iPtr).V1aia)*(Wab_ia_jb+Waa_ia_jb);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m]+=4.0*U(0,e*nels+m);
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "K1Wia";
  nT++; t1=t2;

  U=(*iPtr).V2abia_jb*Waa_ia_jb-(*iPtr).V2aaia_jb*Wab_ia_jb;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+m*nels+m] += 4.0*U(e*nels+m,e*nels+m);
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels && f*nels+n<e*nels+m;n++)
	{
	  x = 2.0*(U(f*nels+n,e*nels+m)+U(e*nels+m,f*nels+n));
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+m*nels+n] += x;
	  grad[n1a+n2aa+(f*virt+e)*T2ab.n_rows+n*nels+m] += x;
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "first cross VW";
  nT++; t1=t2;

  U=(*iPtr).V2abia_jb*Wab_ia_jb-(*iPtr).V2aaia_jb*Waa_ia_jb;
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  x=4.0*(U(e*nels+m,f*nels+n)-U(f*nels+m,e*nels+n)-U(e*nels+n,f*nels+m)+U(f*nels+n,e*nels+m));
	  //E2grad[n1a+n1*T2aa.n_rows+indexD(m,n,nels)] += x;
	  grad[n1a+n1*T2aa.n_rows+indexD(m,n,nels)] += x;
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "2nd cross VW";
  nT++; t1=t2;

  arma::mat Wab_i_jab(nels,nels*virt*virt);
  for(int f=0;f<virt;f++)
  {
    for(int g=0;g<virt;g++)
    {
      n1=f*virt+g;
      for(int m=0;m<nels;m++)
      {
	n2=n1*nels+m;
	for(int i=0;i<nels;i++)
	{
	  Wab_i_jab(i,n2) = T2ab(m*nels+i,n1);
	  //W(i,(f*virt+g)*nels + m) = T2ab(m*nels+i,f*virt+g);
	}
      }
    }
  }
  (*iPtr).E -= 2.0*trace((*iPtr).K1i_j*Wab_i_jab*Wab_i_jab.t());
  (*iPtr).E -= 2.0*trace((*iPtr).K1i_j*T1*T1.t());
  
  (*iPtr).E += 4.0*trace((*iPtr).V2abh_fgn*Wab_i_jab.t()*T1);

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "make Wabijab";
  nT++; t1=t2;

  U=Wab_i_jab*(*iPtr).V2abh_fgn.t();
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m] += 4.0*U(m,e);
    }
  }

  U=(*iPtr).K1i_j*Wab_i_jab;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+(m*nels+m)]-=4.0*U(m,(e*virt+e)*nels+m);
      for(int f=0;f<virt;f++)
      {
	n1=e*virt+f;
	n2=n1*nels+m;
	n3=f*virt+e;
	for(int n=0;n<nels && f*nels+n<e*nels+m;n++)
	{
	  x = U(n,n2) + U(m,n3*nels+n);
	  grad[n1a+n2aa+n1*T2ab.n_rows+m*nels+n] -= 2.0*x;
	  grad[n1a+n2aa+n3*T2ab.n_rows+n*nels+m] -= 2.0*x;
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "Y coming up";
  nT++; t1=t2;

  for(int f=0;f<virt;f++)
  {
    for(int g=0;g<virt;g++)
    {
      n3=f*virt+g;
      n4=g*virt+f;
      for(int m=0;m<nels;m++)
      {
	n1=n3*nels+m;
	Y = Wab_i_jab.col(n1).t()*Wab_i_jab.col(n1);
	//Y = W.col((f*virt+g)*nels+m).t()*W.col((f*virt+g)*nels+m);
	for(int n=0;n<nels;n++)
	{
	  F2abM(m*nels+n,n3) += 2.0*Y(0);
	  F2abM(n*nels+m,n4) += 2.0*Y(0);
	  //F2abM(m*nels+n,f*virt+g) += 2.0*Y(0);
	  //F2abM(n*nels+m,g*virt+f) += 2.0*Y(0);
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "first Y";
  nT++; t1=t2;

  for(int m=0;m<nels;m++)
  {
    Y = Wab_i_jab.row(m)*Wab_i_jab.row(m).t();
    for(int e=0;e<virt;e++)
    {
      F1m(m,e) -= Y(0);
    }
  }

  Wab_i_jab.reset();

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "2nd Y";
  nT++; t1=t2;

  arma::mat Waa_i_jab(nels,nels*virt*(virt-1)/2);
  for(int f=0;f<virt;f++)
  {
    for(int g=f+1;g<virt;g++)
    {
      n2=indexD(f,g,virt);
      for(int m=0;m<nels;m++)
      {
	n3=n2*nels+m;
	Waa_i_jab(m,n3)=0.0;;
	//W(m,indexD(f,g,virt)*nels+m) = 0.0;
	for(int i=0;i<m;i++)
	{
	  Waa_i_jab(i,n3) = -T2aa(indexD(i,m,nels),n2);
	  //W(i,indexD(f,g,virt)*nels+m) = -T2aa(indexD(i,m,nels),indexD(f,g,virt));
	}
	for(int i=m+1;i<nels;i++)
	{
	  Waa_i_jab(i,n3) = T2aa(indexD(m,i,nels),n2);
	  //W(i,indexD(f,g,virt)*nels+m) = T2aa(indexD(m,i,nels),indexD(f,g,virt));
	}
      }
    }
  }
  (*iPtr).E -= 2.0*trace((*iPtr).K1i_j*Waa_i_jab*Waa_i_jab.t());
  (*iPtr).E += 2.0*trace((*iPtr).V2aah_fgn*Waa_i_jab.t()*T1);

  U=(*iPtr).K1i_j*Waa_i_jab;
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  n2=indexD(m,n,nels);
	  //E2agrad[n1a+n1*T2aa.n_rows+n2] += 4.0*(U(m,n1*nels+n)-U(n,n1*nels+m));
	  //E2grad[n1a+n1*T2aa.n_rows+n2] += 4.0*(U(m,n1*nels+n)-U(n,n1*nels+m));
	  grad[n1a+n1*T2aa.n_rows+n2] += 4.0*(U(m,n1*nels+n)-U(n,n1*nels+m));
	}
      }
    }
  }

  U=(*iPtr).V2aah_fgn*Waa_i_jab.t();
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m] += 2.0*U(e,m);
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "make Waa_ijab";
  nT++; t1=t2;

  for(int m=0;m<nels;m++)
  {
    Y = Waa_i_jab.row(m)*Waa_i_jab.row(m).t();
    for(int e=0;e<virt;e++)
    {
      F1m(m,e) -= Y(0);
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n2 = indexD(e,f,virt);
      for(int m=0;m<nels;m++)
      {
	n3=n2*nels+m;
	Y = Waa_i_jab.col(n3).t()*Waa_i_jab.col(n3);
	//Y = W.col(n2*nels+m).t()*W.col(n2*nels+m);
	//Y = W.col(indexD(e,f,virt)*nels+m).t()*W.col(indexD(e,f,virt)*nels+m);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n2) += 2.0*Y(0);
	  //F2aaM(indexD(n,m,nels),indexD(e,f,virt)) += 2.0*Y(0);
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n2) += 2.0*Y(0);
	  //F2aaM(indexD(m,n,nels),indexD(e,f,virt)) += 2.0*Y(0);
	}
      }
    }
  }

  Waa_i_jab.reset();

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "2-3 Y";
  nT++; t1=t2;

  arma::mat Wab_ija_b(nels*nels*virt,virt);
//W_ija_b = T2ab_ij_ba
  for(int i=0;i<nels;i++)
  {
    for(int j=0;j<nels;j++)
    {
      n2=i*nels+j;
      for(int a=0;a<virt;a++)
      {
	n1=n2*virt+a;
	for(int b=0;b<virt;b++)
	{
	  Wab_ija_b(n1,b)=T2ab(n2,b*virt+a);
	  //W((i*nels+j)*virt+b,a)=T2ab(i*nels+j,a*virt+b);
	}
      }
    }
  }
  (*iPtr).E += 2.0*trace((*iPtr).K1a_b*Wab_ija_b.t()*Wab_ija_b);
  (*iPtr).E += 2.0*trace((*iPtr).K1a_b*T1.t()*T1);
  (*iPtr).E -= 4.0*trace((*iPtr).V2abi_jka*Wab_ija_b*T1.t());

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "wabija_b1";
  nT++; t1=t2;

  U=(*iPtr).V2abi_jka*Wab_ija_b;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m] -= 4.0*U(m,e);
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "wabija_b2";
  nT++; t1=t2;

  U=Wab_ija_b*(*iPtr).K1a_b;
  //U=Wab_ija_b*((*iPtr).K1a_b+(*iPtr).V1a_b+(*iPtr).V1aa_b);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+m*nels+m] += 4.0*U((m*nels+m)*virt+e,e);
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels && f*nels+n<e*nels+m;n++)
	{
	  x=2.0*( U((m*nels+n)*virt+f,e)+U((n*nels+m)*virt+e,f));
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+m*nels+n] += x;
	  grad[n1a+n2aa+(f*virt+e)*T2ab.n_rows+n*nels+m] += x;
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "wabija_b3";
  nT++; t1=t2;

  for(int m=0;m<nels;m++)
  {
    for(int n=0;n<nels;n++)
    {
      n2=m*nels+n;
      n3=n*nels+m;
      for(int e=0;e<virt;e++)
      {
	n1=n2*virt+e;
	Y = Wab_ija_b.row(n1)*Wab_ija_b.row(n1).t();
	//Y = W.row((m*nels+n)*virt+e)*W.row((m*nels+n)*virt+e).t();
	for(int f=0;f<virt;f++)
	{
	  F2abM(n2,f*virt+e) += 2.0*Y(0);
	  F2abM(n3,e*virt+f) += 2.0*Y(0);
	  //F2abM(m*nels+n,f*virt+e) += 2.0*Y(0);
	  //F2abM(n*nels+m,e*virt+f) += 2.0*Y(0);
	}
      }
    }
  }

  for(int a=0;a<virt;a++)
  {
    Y = Wab_ija_b.col(a).t()*Wab_ija_b.col(a);
    for(int m=0;m<nels;m++)
    {
      F1m(m,a) -= Y(0);
    }
  }

  Wab_ija_b.reset();

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "5-6 Y";
  nT++; t1=t2;

  arma::mat Waa_ija_b(nels*(nels-1)/2*virt,virt); 
//W_jka_b = T2aa_jk_ab
  for(int j=0;j<nels;j++)
  {
    for(int k=j+1;k<nels;k++)
    {
      n1 = indexD(j,k,nels);
      for(int a=0;a<virt;a++)
      {
	n3=n1*virt+a;
	Waa_ija_b(n3,a)=0.0;
	//W(indexD(j,k,nels)*virt+a,a)=0.0;
	for(int b=a+1;b<virt;b++)
	{
	  x=T2aa(n1,indexD(a,b,virt));
	  Waa_ija_b(n3,b)=-x;
	  Waa_ija_b(n1*virt+b,a)=x;
	  //W(indexD(j,k,nels)*virt+a,b)=-T2aa(indexD(j,k,nels),indexD(a,b,virt));
	  //W(indexD(j,k,nels)*virt+b,a)=T2aa(indexD(j,k,nels),indexD(a,b,virt));
	}
      }
    }
  }
  (*iPtr).E += 2.0*trace((*iPtr).K1a_b*Waa_ija_b.t()*Waa_ija_b);
  (*iPtr).E -= 2.0*trace((*iPtr).V2aai_jka*Waa_ija_b*T1.t());

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "waaija_b";
  nT++; t1=t2;

  U=(*iPtr).K1a_b*Waa_ija_b.t();
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      n1=indexD(m,n,nels);
      for(int e=0;e<virt;e++)
      {
	n2=n1*virt+e;
	for(int f=e+1;f<virt;f++)
	{
	  grad[n1a+indexD(e,f,virt)*T2aa.n_rows+n1] += 4.0*U(e,n1*virt+f);
	  grad[n1a+indexD(e,f,virt)*T2aa.n_rows+n1] -= 4.0*U(f,n2);
	}
      }
    }
  }
  U=(*iPtr).V2aai_jka*Waa_ija_b;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m] -= 2.0*U(m,e);
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "waaija_b";
  nT++; t1=t2;

  for(int e=0;e<virt;e++)
  {
    Y = Waa_ija_b.col(e).t()*Waa_ija_b.col(e);
    for(int m=0;m<nels;m++)
    {
      F1m(m,e) -= Y(0);
    }
  }
  for(int i=0;i<nels;i++)
  {
    for(int j=i+1;j<nels;j++)
    {
      n1 = indexD(i,j,nels);
      for(int a=0;a<virt;a++)
      {
	n2=n1*virt+a;
	Y = Waa_ija_b.row(n2)*Waa_ija_b.row(n2).t();
	//Y = W.row(indexD(i,j,nels)*virt+a)*W.row(indexD(i,j,nels)*virt+a).t();
	for(int b=0;b<a;b++)
	{
	  F2aaM(n1,indexD(b,a,virt)) += 2.0*Y(0);
	  //F2aaM(indexD(i,j,nels),indexD(b,a,virt)) += 2.0*Y(0);
	}
	for(int b=a+1;b<virt;b++)
	{
	  F2aaM(n1,indexD(a,b,virt)) += 2.0*Y(0);
	  //F2aaM(indexD(i,j,nels),indexD(a,b,virt)) += 2.0*Y(0);
	}
      }
    }
  }
  Waa_ija_b.reset();

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "Fs done";
  nT++; t1=t2;

  (*iPtr).E += 4.0*trace((*iPtr).K1ia*arma::vectorise(T1%sqrt(F1m)));
  (*iPtr).E += 4.0*trace((*iPtr).K1ia*(Wab_ia_jb+Waa_ia_jb)*vectorise(T1));

  Wab_ia_jb.reset();
  Waa_ia_jb.reset();

  (*iPtr).E += 2.0*accu((*iPtr).V2abij_ab%T2ab%sqrt(F2abM));
  (*iPtr).E += 2.0*accu((*iPtr).V2aaij_ab%T2aa%sqrt(F2aaM));
 
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "accus";
  nT++; t1=t2;
 
  U=(*iPtr).K1i_j*T1;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m]-=4.0*U(m,e);
    }
  }
  U=(*iPtr).K1a_b*T1.t();
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m]+=4.0*U(e,m);
    }
  }
  U=((*iPtr).V2abia_jb-(*iPtr).V2aaia_jb)*arma::vectorise(T1);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m] += 4.0*U(e*nels+m);
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "blah";
  nT++; t1=t2;

  U=(*iPtr).V2aaij_kl*T2aa;
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  grad[n1a+n1*T2aa.n_rows+indexD(m,n,nels)] += 2.0*U(indexD(m,n,nels),n1);
	}
      }
    }
  }

  U = T2aa*(*iPtr).V2aaab_cd;
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  n2=indexD(m,n,nels);
	  grad[n1a+n1*T2aa.n_rows+n2] += 2.0*U(n2,n1);
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "1U done";
  nT++; t1=t2;

  U=T1*(*iPtr).V2aah_fgn;
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      for(int m=0;m<nels;m++)
      {
	n2=n1*nels+m;
	for(int n=m+1;n<nels;n++)
	{
	  grad[n1a+n1*T2aa.n_rows+indexD(m,n,nels)] += 2.0*(U(n,n2)-U(m,n1*nels+n));
	}
      }
    }
  }

  U=T1.t()*(*iPtr).V2aai_jka;
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      n1=indexD(m,n,nels);
      for(int e=0;e<virt;e++)
      {
	n2=n1*virt+e;
	for(int f=e+1;f<virt;f++)
	{
	  n3=indexD(e,f,virt);
	  grad[n1a+n3*T2aa.n_rows+n1] += 2.0*(U(f,n2)-U(e,n1*virt+f));
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "2U done";
  nT++; t1=t2;

  U=(*iPtr).V2abij_kl*T2ab;
  for(int e=0;e<virt;e++)
  {
    for(int k=0;k<nels;k++)
    {
      grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+(k*nels+k)] += 2.0*U(k*nels+k,e*virt+e);
      for(int f=0;f<virt;f++)
      {
	for(int l=0;l<nels && f*nels+l<e*nels+k;l++)
	{
	  x=2.0*U(k*nels+l,e*virt+f);
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+(k*nels+l)] += x;
	  grad[n1a+n2aa+(f*virt+e)*T2ab.n_rows+(l*nels+k)] += x;
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "a3U done";
  nT++; t1=t2;

  U=T2ab*(*iPtr).V2abab_cd;

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "multiplied";
  nT++; t1=t2;

  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+m*nels+m] += 2.0*U(m*nels+m,e*virt+e); 
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels && f*nels+n<e*nels+m;n++)
	{
	  x=2.0*U(m*nels+n,e*virt+f);
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+m*nels+n] += x;
	  grad[n1a+n2aa+(f*virt+e)*T2ab.n_rows+n*nels+m] += x; 
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "3U done";
  nT++; t1=t2;

  U=T1*(*iPtr).V2abh_fgn;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+m*nels+m] += 4.0*U(m,(e*virt+e)*nels+m);
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels && f*nels+n<e*nels+m;n++)
	{
	  x = 2.0*(U(n,(e*virt+f)*nels+m)+U(m,(f*virt+e)*nels+n));
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+m*nels+n] += x; 
	  grad[n1a+n2aa+(f*virt+e)*T2ab.n_rows+n*nels+m] += x; 
	}
      }
    }
  }

  U=T1.t()*(*iPtr).V2abi_jka;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+m*nels+m] -= 4.0*U(e,(m*nels+m)*virt+e);
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels && f*nels+n<e*nels+m;n++)
	{
	  x = 2.0*(U(f,(n*nels+m)*virt+e)+U(e,(m*nels+n)*virt+f));
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+m*nels+n] -= x;
	  grad[n1a+n2aa+(f*virt+e)*T2ab.n_rows+n*nels+m] -= x; 
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "Us done";
  nT++; t1=t2;

  arma::mat P1(nels,virt,arma::fill::zeros);
  arma::mat P2ab(nels*nels,virt*virt,arma::fill::zeros);
  arma::mat P2aa(nels*(nels-1)/2,virt*(virt-1)/2,arma::fill::zeros);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      P1(m,e)=4.0*(*iPtr).K1ia(e*nels+m)*T1(m,e)/sqrt(F1m(m,e));
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      x=0.0;
      for(int f=0;f<virt;f++)
      {
        n1=e*virt+f;
        n2=(e+nels)*nact+f+nels;
	for(int n=0;n<nels;n++)
	{
	  P2ab(m*nels+n,n1) = 2.0*(*iPtr).V2abij_ab(m*nels+n,e*virt+f)*T2ab(m*nels+n,n1)/sqrt(F2abM(m*nels+n,n1));
	}
      }
    }
  }

  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      n2=indexD(e+nels,f+nels,nact);
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  n3=indexD(m,n,nels);
	  n4=indexD(m,n,nact);
	  P2aa(n3,n1) = 2.0*(*iPtr).V2aaij_ab(n3,n1)*T2aa(n3,n1)/sqrt(F2aaM(n3,n1));
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "Ps done";
  nT++; t1=t2;

  arma::mat Q1(nels,virt,arma::fill::zeros);
  arma::mat Q2aa(nels*(nels-1)/2,virt*(virt-1)/2,arma::fill::zeros);
  arma::mat Q2ab(nels*nels,virt*virt,arma::fill::zeros);

  Q1=P1;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      x=0.0;
      for(int a=0;a<virt;a++)
      {
	x += P1(m,a);
      }
      for(int i=0;i<nels;i++)
      {
	x += P1(i,e);
      }
      Q1(m,e) -= x;
      for(int a=0;a<e;a++)
      {
	for(int i=0;i<m;i++)
	{
	  Q2aa(indexD(i,m,nels),indexD(a,e,virt)) -= x;
	}
	for(int i=m+1;i<nels;i++)
	{
	  Q2aa(indexD(m,i,nels),indexD(a,e,virt)) -= x;
	}
      }
      for(int a=e+1;a<virt;a++)
      {
	for(int i=0;i<m;i++)
	{
	  Q2aa(indexD(i,m,nels),indexD(e,a,virt)) -= x;
	}
	for(int i=m+1;i<nels;i++)
	{
	  Q2aa(indexD(m,i,nels),indexD(e,a,virt)) -= x;
	}
      }
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      x = 0.0;
      for(int a=0;a<e;a++)
      {
	for(int i=0;i<m;i++)
	{
	  x += P2aa(indexD(i,m,nels),indexD(a,e,virt));
	}
	for(int i=m+1;i<nels;i++)
	{
	  x += P2aa(indexD(m,i,nels),indexD(a,e,virt));
	}
      }
      for(int a=e+1;a<virt;a++)
      {
	for(int i=0;i<m;i++)
	{
	  x += P2aa(indexD(i,m,nels),indexD(e,a,virt));
	}
	for(int i=m+1;i<nels;i++)
	{
	  x += P2aa(indexD(m,i,nels),indexD(e,a,virt));
	}
      }
      Q1(m,e) += x;
      for(int a=0;a<e;a++)
      {
	for(int i=0;i<m;i++)
	{
	  Q2aa(indexD(i,m,nels),indexD(a,e,virt)) -= x;
	}
	for(int i=m+1;i<nels;i++)
	{
	  Q2aa(indexD(m,i,nels),indexD(a,e,virt)) -= x;
	}
      }
      for(int a=e+1;a<virt;a++)
      {
	for(int i=0;i<m;i++)
	{
	  Q2aa(indexD(i,m,nels),indexD(e,a,virt)) -= x;
	}
	for(int i=m+1;i<nels;i++)
	{
	  Q2aa(indexD(m,i,nels),indexD(e,a,virt)) -= x;
	}
      }
      x *= 0.5;
      for(int i=0;i<nels;i++)
      {
	Q1(i,e) -= x;
      }
      for(int a=0;a<virt;a++)
      {
	Q1(m,a) -= x;
      }
      for(int a=0;a<virt;a++)
      {
	for(int i=0;i<nels;i++)
	{
	  Q2ab(i*nels+m,a*virt+e) -= x;
	  Q2ab(m*nels+i,e*virt+a) -= x;
	}
      }
    }
  }

  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      x=0.0;
      for(int i=0;i<nels;i++)
      {
	for(int j=i+1;j<nels;j++)
	{
	  x += P2aa(indexD(i,j,nels),indexD(e,f,virt));
	}
      }
      for(int i=0;i<nels;i++)
      {
	for(int j=i+1;j<nels;j++)
	{
	  Q2aa(indexD(i,j,nels),indexD(e,f,virt)) -= x;
	}
      }
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      for(int m=0;m<nels;m++)
      {
	x=0.0;
	for(int i=0;i<m;i++)
	{
	  x += P2aa(indexD(i,m,nels),indexD(e,f,virt));
	}
	for(int i=m+1;i<nels;i++)
	{
	  x += P2aa(indexD(m,i,nels),indexD(e,f,virt));
	}
	x *= 2.0;
	for(int i=0;i<m;i++)
	{
	  Q2aa(indexD(i,m,nels),indexD(e,f,virt)) += x;
	}
	for(int i=m+1;i<nels;i++)
	{
	  Q2aa(indexD(m,i,nels),indexD(e,f,virt)) += x;
	}
      }
    }
  }
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      x=0.0;
      for(int a=0;a<virt;a++)
      {
	for(int b=a+1;b<virt;b++)
	{
	  x += P2aa(indexD(m,n,nels),indexD(a,b,virt));
	}
      }
      for(int a=0;a<virt;a++)
      {
	for(int b=a+1;b<virt;b++)
	{
	  Q2aa(indexD(m,n,nels),indexD(a,b,virt)) -= x;
	}
      }
    }
  }
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      for(int e=0;e<virt;e++)
      {
	x=0.0;
	for(int a=0;a<e;a++)
	{
	  x += P2aa(indexD(m,n,nels),indexD(a,e,virt));
	}
	for(int a=e+1;a<virt;a++)
	{
	  x += P2aa(indexD(m,n,nels),indexD(e,a,virt));
	}
	x *= 2.0;
	for(int a=0;a<e;a++)
	{
	  Q2aa(indexD(m,n,nels),indexD(a,e,virt)) += x;
	}
	for(int a=e+1;a<virt;a++)
	{
	  Q2aa(indexD(m,n,nels),indexD(e,a,virt)) += x;
	}
      }
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      x=0.0;
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels;n++)
	{
	  x += P2ab(m*nels+n,e*virt+f);
	}
      }
      for(int a=0;a<virt;a++)
      {
	for(int i=0;i<nels;i++)
	{
	  Q2ab(m*nels+i,e*virt+a) -= x;
	  Q2ab(i*nels+m,a*virt+e) -= x;
	}
      }
      x *= 2.0;
      Q1(m,e) += x;
      for(int a=0;a<virt;a++)
      {
	Q1(m,a) -= x;
      }
      for(int i=0;i<nels;i++)
      {
	Q1(i,e) -= x;
      }
      for(int a=0;a<e;a++)
      {
	for(int i=0;i<m;i++)
	{
	  Q2aa(indexD(i,m,nels),indexD(a,e,virt)) -= x;
	}
	for(int i=m+1;i<nels;i++)
	{
	  Q2aa(indexD(m,i,nels),indexD(a,e,virt)) -= x;
	}
      }
      for(int a=e+1;a<virt;a++)
      {
	for(int i=0;i<m;i++)
	{
	  Q2aa(indexD(i,m,nels),indexD(e,a,virt)) -= x;
	}
	for(int i=m+1;i<nels;i++)
	{
	  Q2aa(indexD(m,i,nels),indexD(e,a,virt)) -= x;
	}
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "Q12aa done";
  nT++; t1=t2;

  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {
      for(int m=0;m<nels;m++)
      {
	x=0.0;
	for(int i=0;i<nels;i++)
	{
	  x += P2ab(m*nels+i,e*virt+f);
	}
	x *= 2.0;
	for(int i=0;i<nels;i++)
	{
	  Q2ab(m*nels+i,e*virt+f) += x;
	  Q2ab(i*nels+m,f*virt+e) += x;
	}
      }
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      for(int n=0;n<nels;n++)
      {
	x=0.0;
	for(int a=0;a<virt;a++)
	{
	  x += P2ab(m*nels+n,e*virt+a);
	}
	x *= 2.0;
	for(int a=0;a<virt;a++)
	{
	  Q2ab(m*nels+n,e*virt+a) += x;
	  Q2ab(n*nels+m,a*virt+e) += x;
	}
      }
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {
      x=0.0;
      for(int i=0;i<nels;i++)
      {
	for(int j=0;j<nels;j++)
	{
	  x += P2ab(i*nels+j,e*virt+f);
	}
      }
      for(int i=0;i<nels;i++)
      {
	for(int j=0;j<nels;j++)
	{
	  Q2ab(i*nels+j,e*virt+f) -= x;
	}
      }
    }
  }
  for(int m=0;m<nels;m++)
  {
    for(int n=0;n<nels;n++)
    {
      x=0.0;
      for(int a=0;a<virt;a++)
      {
	for(int b=0;b<virt;b++)
	{
	  x += P2ab(m*nels+n,a*virt+b);
	}
      }
      for(int a=0;a<virt;a++)
      {
	for(int b=0;b<virt;b++)
	{
	  Q2ab(m*nels+n,a*virt+b) -= x;
	}
      }
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      x=0.0;
      for(int a=0;a<virt;a++)
      {
	for(int i=0;i<nels;i++)
	{
	  x += P2ab(i*nels+m,e*virt+a);
	}
      }
      for(int a=0;a<virt;a++)
      {
	for(int i=0;i<nels;i++)
	{
	  Q2ab(i*nels+m,e*virt+a) -= x;
	  Q2ab(m*nels+i,a*virt+e) -= x;
	}
      }
    }
  }

  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  Q2aa(indexD(m,n,nels),indexD(e,f,virt)) += P1(m,e)+P1(m,f)+P1(n,e)+P1(n,f);
	  Q2aa(indexD(m,n,nels),indexD(e,f,virt)) -= 3.0*P2aa(indexD(m,n,nels),indexD(e,f,virt));
	}
      }
    }
  }
  P2aa.reset();
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {
      for(int m=0;m<nels;m++)
      {
	for(int n=0;n<nels;n++)
	{
	  Q2ab(m*nels+n,e*virt+f) += P1(m,e)+P1(n,f);
	  Q2ab(m*nels+n,e*virt+f) -= 3.0*P2ab(m*nels+n,e*virt+f);
	}
      }
    }
  }
  P1.reset();
  P2ab.reset();

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "Q2ab done";
  nT++; t1=t2;

  Q1 = Q1%T1;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m] += Q1(m,e);
    }
  }
  Q1.reset();
  Q2aa = Q2aa%T2aa;
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  grad[n1a+n1*T2aa.n_rows+indexD(m,n,nels)] += Q2aa(indexD(m,n,nels),n1);
	}
      }
    }
  }
  Q2aa.reset();
  Q2ab = Q2ab%T2ab;
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {
      for(int m=0;m<nels;m++)
      {
	for(int n=0;n<nels;n++)
	{
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+(m*nels+n)] += Q2ab(m*nels+n,e*virt+f);
	}
      }
    }
  }
  Q2ab.reset();

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "Qs added";
  nT++; t1=t2;

  double maxT1=0.0;
  double maxT2aa=0.0;
  double maxT2ab=0.0;
  for(int a=0;a<virt;a++)
  {
    for(int i=0;i<nels;i++)
    {
      x=4.0*(*iPtr).K1ia(a*nels+i)*sqrt(F1m(i,a));
      grad[a*nels+i]+=x;
      if(fabs(grad[a*nels+i]) > maxT1)
      {
	maxT1 = fabs(grad[a*nels+i]);
      }
    }
  }

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "grad1 finished";
  nT++; t1=t2;

  for(int a=0;a<virt;a++)
  {
    n1=indexD1(a,virt);
    n2=indexD1(a+nels,nact);
    for(int b=a+1;b<virt;b++)
    {
      n3=n1+b;
      n4=n2+b+nels;
      for(int i=0;i<nels;i++)
      {
	n5=indexD1(i,nels);
	n6=indexD1(i,nact);
        for(int j=i+1;j<nels;j++)
	{
	  x=4.0*((*iPtr).K1ia(a*nels+i)*T1(j,b)-(*iPtr).K1ia(a*nels+j)*T1(i,b)-(*iPtr).K1ia(b*nels+i)*T1(j,a)+(*iPtr).K1ia(b*nels+j)*T1(i,a));
	  x+= 2.0*(*iPtr).V2aaij_ab(indexD(i,j,nels),indexD(a,b,virt))*sqrt(F2aaM(n5+j,n3));
	  grad[n1a+n3*T2aa.n_rows+n5+j]+=x;
	  if(fabs(grad[n1a+n3*T2aa.n_rows+n5+j]) > maxT2aa)
	  {
	    maxT2aa = fabs(grad[n1a+n3*T2aa.n_rows+n5+j]  );
 	  }
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "grad2aa finished";
  nT++; t1=t2;

  for(int a=0;a<virt;a++)
  {
    for(int i=0;i<nels;i++)
    {
      y=4.0*(*iPtr).K1ia(a*nels+i)*T1(i,a);
      y += 2.0*(*iPtr).V2abij_ab(i*nels+i,a*virt+a)*sqrt(F2abM(i*nels+i,a*virt+a));
      grad[n1a+n2aa+(a*virt+a)*T2ab.n_rows+(i*nels+i)]+=y;
      if(fabs(grad[n1a+n2aa+(a*virt+a)*T2ab.n_rows+i*nels+i]  ) > maxT2ab)
      {
	maxT2ab = fabs(grad[n1a+n2aa+(a*virt+a)*T2ab.n_rows+i*nels+i]  );
      }
    }
  }
  y=0.0;
  for(int a=0;a<virt;a++)
  {
    for(int b=0;b<virt;b++)
    {
      for(int i=0;i<nels;i++)
      {
	for(int j=0;j<nels && (b*nels+j)<a*nels+i;j++)
	{
	  y=2.0*((*iPtr).K1ia(a*nels+i)*T1(j,b)+(*iPtr).K1ia(b*nels+j)*T1(i,a));
          y += 2.0*(*iPtr).V2abij_ab(i*nels+j,a*virt+b)*sqrt(F2abM(i*nels+j,a*virt+b));
	  grad[n1a+n2aa+(a*virt+b)*T2ab.n_rows+(i*nels+j)]+=y;
	  grad[n1a+n2aa+(b*virt+a)*T2ab.n_rows+(j*nels+i)]+=y;
	  if(fabs(grad[n1a+n2aa+(a*virt+b)*T2ab.n_rows+i*nels+j]  ) > maxT2ab)
	  {
	    maxT2ab = fabs(grad[n1a+n2aa+(a*virt+b)*T2ab.n_rows+i*nels+j]  );
	  }
	}
      }
    }
  }
  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "grad2ab finished";
  nT++; t1=t2;

  (*iPtr).E += trace((*iPtr).V2abij_kl.t()*T2ab*T2ab.t());
  (*iPtr).E += trace((*iPtr).V2abab_cd.t()*T2ab.t()*T2ab);

  E=(*iPtr).E;

  setTime(t1,t2,time_span);
  (*iPtr).ts[nT] += time_span;
  (*iPtr).tLabel[nT] = "E finished";
  nT++; t1=t2;

//  std::cout << "\tE:\t" << (*iPtr).E+(*iPtr).Enuc;
//  std::cout << "\tmax dT2ab:\t" << maxT2ab;
//  std::cout << "\tmax dT2aa:\t" << maxT2aa;
//  std::cout << "\tmax dT1:\t" << maxT1 << '\t';
//  std::cout << std::endl;

  (*iPtr).maxT=maxT1;
  if(maxT2aa > (*iPtr).maxT)
  {
    (*iPtr).maxT = maxT2aa;
  }
  if(maxT2ab > (*iPtr).maxT)
  {
    (*iPtr).maxT = maxT2ab;
  }

  return;
}

void setMats(intBundle &i)
{
  int nels=i.nels;
  int nact=0;
  for(int p=0;p<i.K1s.size();p++)
  {
    nact += i.K1s[p].n_cols;
  }
  i.virt=nact-nels;
  int virt=i.virt;
  double x;
  int n1,n2,n3;
  i.Ehf=0.0;

  std::vector<int> nOcc=i.nOcc;
  std::vector<int> nVirt;
  std::vector<int> nTot;
  for(int p=0;p<nOcc.size();p++)
  {
    nVirt.push_back(i.K1s[p].n_cols-nOcc[p]);
    nTot.push_back(i.K1s[p].n_cols);
  }

  int o1,o2,o3,o4;
  int v1,v2,v3,v4;
  i.K1i_j.set_size(nels,nels);
  i.K1i_j.zeros();
  o1=0;
  for(int p=0;p<i.K1s.size();p++)
  {
    for(int m=0;m<i.nOcc[p];m++)
    {
      for(int n=0;n<i.nOcc[p];n++)
      {
	i.K1i_j(o1+m,o1+n)=i.K1s[p](m,n);
      }
    }
    o1 += i.nOcc[p];
  }
  i.K1a_b.set_size(virt,virt); i.K1a_b.zeros();
  v1=0;
  for(int p=0;p<i.K1s.size();p++)
  {
    for(int a=0;a<nVirt[p];a++)
    {
      for(int b=0;b<nVirt[p];b++)
      {
	i.K1a_b(v1+b,v1+a)=i.K1s[p](b+nOcc[p],a+nOcc[p]);
      }
    }
    v1 += nVirt[p];
  }
  i.K1ia.set_size(1,nels*virt); i.K1ia.zeros();
  o1=0;v1=0;
  for(int p=0;p<i.K1s.size();p++)
  {
    for(int a=0;a<nVirt[p];a++)
    {
      for(int m=0;m<nOcc[p];m++)
      {
	i.K1ia(0,(a+v1)*nels+m+o1)=i.K1s[p](m,a+nOcc[p]);
      }
    }
    o1 += nOcc[p];
    v1 += nVirt[p];
  }
  for(int m=0;m<nels;m++)
  {
    i.Ehf += 2.0*i.K1i_j(m,m);
  }
  o1=0;
  for(int p=0;p<i.K1s.size();p++)
  {
    for(int m=0;m<nOcc[p];m++)
    {
      for(int n=0;n<nOcc[p];n++)
      {
	x=0;
	for(int q=0;q<i.K1s.size();q++)
	{
	  for(int t=0;t<nOcc[q];t++)
	  {
	    x += 2.0*i.V2s[p*i.K1s.size()+q][p*i.K1s.size()+q](n*nTot[q]+t,m*nTot[q]+t);
	    x -= i.V2s[q*i.K1s.size()+p][p*i.K1s.size()+q](t*nTot[p]+n,m*nTot[q]+t);
	  }
	}
	i.K1i_j(o1+n,o1+m) += x;
      }
    }
    o1 += nOcc[p];
  }
  o1=0; v1=0;
  for(int p=0;p<nOcc.size();p++)
  {
    for(int e=0;e<nVirt[p];e++)
    {
      for(int m=0;m<nOcc[p];m++)
      {
	x=0.0;
	for(int q=0;q<nOcc.size();q++)
	{
	  for(int n=0;n<nOcc[q];n++)
	  {
	    x += 2.0*i.V2s[p*nOcc.size()+q][p*nOcc.size()+q](m*nTot[q]+n,(e+nOcc[p])*nTot[q]+n);
	    x -= i.V2s[q*nOcc.size()+p][p*nOcc.size()+q](n*nTot[p]+m,(e+nOcc[p])*nTot[q]+n);
	  }
	}
	i.K1ia(0,(e+v1)*nels+m+o1) += x;
      }
    }
    o1 += nOcc[p];
    v1 += nVirt[p];
  }
  i.V2abi_jka.set_size(nels,nels*nels*virt);
  o1=0;
  for(int p=0;p<nOcc.size();p++)
  {
    o2=0;
    for(int q=0;q<nOcc.size();q++)
    {
      v3=0;
      for(int r=0;r<nOcc.size();r++)
      {
	o4=0;
	for(int s=0;s<nOcc.size();s++)
	{
	  if(i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_rows>0 && i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_cols>0)
	  {
	    for(int j=0;j<nOcc[p];j++)
	    {
	      for(int k=0;k<nOcc[q];k++)
	      {
		for(int e=0;e<nVirt[r];e++)
		{
		  for(int l=0;l<nOcc[s];l++)
		  {
		    i.V2abi_jka(l+o4,((j+o1)*nels+k+o2)*virt+e+v3) = i.V2s[s*nOcc.size()+r][p*nOcc.size()+q](l*nTot[r]+e+nOcc[r],j*nTot[q]+k);
		  }
		}
	      }
	    }
	  }
	  else
	  {
	    for(int j=0;j<nOcc[p];j++)
	    {
	      for(int k=0;k<nOcc[q];k++)
	      {
		for(int e=0;e<nVirt[r];e++)
		{
		  for(int l=0;l<nOcc[s];l++)
		  {
		    i.V2abi_jka(l+o4,((j+o1)*nels+k+o2)*virt+e+v3) = 0.0;
		  }
		}
	      }
	    }
	  }
	  o4 += nOcc[s];
	}
	v3 += nVirt[r];
      }
      o2 += nOcc[q];
    }
    o1 += nOcc[p];
  }
  i.V2aai_jka.resize(nels,nels*(nels-1)/2*virt);
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      n1=indexD(m,n,nact);
      n3=indexD(m,n,nels);
      for(int e=0;e<virt;e++)
      {
	n2=n3*virt+e;
	for(int k=0;k<nels;k++)
	{
	  i.V2aai_jka(k,n2) = 2.0*(i.V2abi_jka(k,(m*nels+n)*virt+e)-i.V2abi_jka(k,(n*nels+m)*virt+e));
	}
      }
    }
  }
  v1=0;
  for(int p=0;p<nOcc.size();p++)
  {
    for(int e=0;e<nVirt[p];e++)
    {
      for(int f=0;f<nVirt[p];f++)
      {
	x=0;
	o1=0;
	for(int q=0;q<nOcc.size();q++)
	{
	  for(int m=0;m<nOcc[q];m++)
	  {
	    x += 2.0*i.V2s[p*nOcc.size()+q][p*nOcc.size()+q]((f+nOcc[p])*nTot[q]+m,(e+nOcc[p])*nTot[q]+m);
	    x -= i.V2s[q*nOcc.size()+p][p*nOcc.size()+q]((m)*nTot[p]+f+nOcc[p],(e+nOcc[p])*nTot[q]+m);
	  }
	  o1 += nOcc[q];
	}
	i.K1a_b(f+v1,e+v1) += x;
      }
    }
    v1 += nVirt[p];
  }
  i.V2abia_jb.set_size(nels*virt,nels*virt);
  v1=0;
  for(int p=0;p<nOcc.size();p++)
  {
    o2=0;
    for(int q=0;q<nOcc.size();q++)
    {
      v3=0;
      for(int r=0;r<nOcc.size();r++)
      {
	o4=0;
	for(int s=0;s<nOcc.size();s++)
	{
	  if(i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_rows>0 && i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_cols>0)
	  {
	    for(int b=0;b<nVirt[p];b++)
	    {
	      for(int j=0;j<nOcc[q];j++)
	      {
		for(int a=0;a<nVirt[r];a++)
		{
		  for(int k=0;k<nOcc[s];k++)
		  {
		    i.V2abia_jb((a+v3)*nels+k+o4,(b+v1)*nels+j+o2) = i.V2s[r*nOcc.size()+q][s*nOcc.size()+p]((a+nOcc[r])*nTot[q]+j,k*nTot[p]+b+nOcc[p]);
		  }
		}
	      }
	    }
	  }
	  else
	  {
	    for(int b=0;b<nVirt[p];b++)
	    {
	      for(int j=0;j<nOcc[q];j++)
	      {
		for(int a=0;a<nVirt[r];a++)
		{
		  for(int k=0;k<nOcc[s];k++)
		  {
		    i.V2abia_jb((a+v3)*nels+k+o4,(b+v1)*nels+j+o2) = 0;
		  }
		}
	      }
	    }
	  }
	  o4 += nOcc[s];
	}
	v3 += nVirt[r];
      }
      o2 += nOcc[q];
    }
    v1 += nVirt[p];
  }
  i.V2abia_jb2.set_size(nels*virt,nels*virt);
  v1=0;
//emfn
  for(int p=0;p<nOcc.size();p++)
  {
    o2=0;
    for(int q=0;q<nOcc.size();q++)
    {
      v3=0;
      for(int r=0;r<nOcc.size();r++)
      {
	o4=0;
	for(int s=0;s<nOcc.size();s++)
	{
	  if(i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_rows>0 && i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_cols>0)
	  {
	    for(int b=0;b<nVirt[p];b++)
	    {
	      for(int j=0;j<nOcc[q];j++)
	      {
		for(int a=0;a<nVirt[r];a++)
		{
		  for(int k=0;k<nOcc[s];k++)
		  {
		    i.V2abia_jb2((a+v3)*nels+k+o4,(b+v1)*nels+j+o2) = i.V2s[q*nOcc.size()+r][s*nOcc.size()+p](j*nTot[r]+a+nOcc[r],k*nTot[p]+b+nOcc[p]);
		  }
		}
	      }
	    }
	  }
	  else
	  {
	    for(int b=0;b<nVirt[p];b++)
	    {
	      for(int j=0;j<nOcc[q];j++)
	      {
		for(int a=0;a<nVirt[r];a++)
		{
		  for(int k=0;k<nOcc[s];k++)
		  {
		    i.V2abia_jb2((a+v3)*nels+k+o4,(b+v1)*nels+j+o2) = 0.0;
		  }
		}
	      }
	    }
	  }
	  o4 += nOcc[s];
	}
	v3 += nVirt[r];
      }
      o2 += nOcc[q];
    }
    v1 += nVirt[p];
  }
  i.V2aaia_jb.set_size(nels*virt,nels*virt);
  for(int b=0;b<virt;b++)
  {
    for(int j=0;j<nels;j++)
    {
      for(int a=0;a<virt;a++)
      {
	for(int k=0;k<nels;k++)
	{
	  i.V2aaia_jb(b*nels+j,a*nels+k) = i.V2abia_jb2(b*nels+j,a*nels+k)-i.V2abia_jb(b*nels+j,a*nels+k);	  
	  //i.V2aaia_jb(b*nels+j,a*nels+k) = 0.5*i.V2aa(indexD(j,a+nels,nact),indexD(k,b+nels,nact));	  
	}
      }
    }
  }
  i.V2abh_fgn.set_size(virt,virt*virt*nels);
  v1=0;
  for(int p=0;p<nOcc.size();p++)
  {
    v2=0;
    for(int q=0;q<nOcc.size();q++)
    {
      o3=0;
      for(int r=0;r<nOcc.size();r++)
      {
	v4=0;
	for(int s=0;s<nOcc.size();s++)
	{
	  if(i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_rows>0 && i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_cols>0)
	  {
	    for(int f=0;f<nVirt[p];f++)
	    {
	      for(int g=0;g<nVirt[q];g++)
	      {
		for(int n=0;n<nOcc[r];n++)
		{
		  for(int h=0;h<nVirt[s];h++)
		  {
		    i.V2abh_fgn(h+v4,((f+v1)*virt+g+v2)*nels+n+o3) = i.V2s[r*nOcc.size()+s][p*nOcc.size()+q](n*nTot[s]+h+nOcc[s],(f+nOcc[p])*nTot[q]+g+nOcc[q]);
		  }
		}
	      }
	    }
	  }
	  else
	  {
	    for(int f=0;f<nVirt[p];f++)
	    {
	      for(int g=0;g<nVirt[q];g++)
	      {
		for(int n=0;n<nOcc[r];n++)
		{
		  for(int h=0;h<nVirt[s];h++)
		  {
		    i.V2abh_fgn(h+v4,((f+v1)*virt+g+v2)*nels+n+o3) = 0;
		  }
		}
	      }
	    }
	  }
	  v4 += nVirt[s];
	}
	o3 += nOcc[r];
      }
      v2 += nVirt[q];
    }
    v1 += nVirt[p];
  }
  i.V2aah_fgn.set_size(virt,virt*(virt-1)/2*nels);
  for(int f=0;f<virt;f++)
  {
    for(int g=f+1;g<virt;g++)
    {
      n1=indexD(f,g,virt);
      n3=indexD(f+nels,g+nels,nact);
      for(int m=0;m<nels;m++)
      {
	n2=n1*nels+m;
	for(int h=0;h<virt;h++)
	{
	  i.V2aah_fgn(h,n2) = 2.0*(i.V2abh_fgn(h,(f*virt+g)*nels+m)-i.V2abh_fgn(h,(g*virt+f)*nels+m));
	}
      }
    }
  }
  i.V2abij_kl.set_size(nels*nels,nels*nels);
  o1=0;
  for(int p=0;p<nOcc.size();p++)
  {
    o2=0;
    for(int q=0;q<nOcc.size();q++)
    {
      o3=0;
      for(int r=0;r<nOcc.size();r++)
      {
	o4=0;
	for(int s=0;s<nOcc.size();s++)
	{
	  if(i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_rows>0 && i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_cols>0)
	  {
	    for(int m=0;m<nOcc[p];m++)
	    {
	      for(int j=0;j<nOcc[q];j++)
	      {
		for(int k=0;k<nOcc[r];k++)
		{
		  for(int l=0;l<nOcc[s];l++)
		  {
		    i.V2abij_kl((k+o3)*nels+l+o4,(m+o1)*nels+j+o2) = i.V2s[r*nOcc.size()+s][p*nOcc.size()+q](k*nTot[s]+l,m*nTot[q]+j);
		  }
		}
	      }
	    }
	  }
	  else
	  {
	    for(int m=0;m<nOcc[p];m++)
	    {
	      for(int j=0;j<nOcc[q];j++)
	      {
		for(int k=0;k<nOcc[r];k++)
		{
		  for(int l=0;l<nOcc[s];l++)
		  {
		    i.V2abij_kl((k+o3)*nels+l+o4,(m+o1)*nels+j+o2) = 0;
		  }
		}
	      }
	    }
	  }
	  o4 += nOcc[s];
	}
	o3 += nOcc[r];
      }
      o2 += nOcc[q];
    }
    o1 += nOcc[p];
  }
  for(int m=0;m<nels;m++)
  {
    for(int n=0;n<nels;n++)
    {
      i.Ehf += 2.0*i.V2abij_kl(m*nels+n,m*nels+n)-i.V2abij_kl(n*nels+m,m*nels+n);
    }
  }
  i.V2aaij_kl.set_size(nels*(nels-1)/2,nels*(nels-1)/2);
  for(int j=0;j<nels;j++)
  {
    for(int k=j+1;k<nels;k++)
    {
      for(int l=0;l<nels;l++)
      {
	for(int m=l+1;m<nels;m++)
	{
	  i.V2aaij_kl(indexD(l,m,nels),indexD(j,k,nels)) = 2.0*(i.V2abij_kl(l*nels+m,j*nels+k)-i.V2abij_kl(m*nels+l,j*nels+k));
	}
      }
    }
  }
  i.V2abab_cd.set_size(virt*virt,virt*virt);
  v1=0;
  for(int p=0;p<nOcc.size();p++)
  {
    v2=0;
    for(int q=0;q<nOcc.size();q++)
    {
      v3=0;
      for(int r=0;r<nOcc.size();r++)
      {
	v4=0;
	for(int s=0;s<nOcc.size();s++)
	{
	  if(i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_rows>0 && i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_cols>0)
	  {
	    for(int e=0;e<nVirt[p];e++)
	    {
	      for(int f=0;f<nVirt[q];f++)
	      {
		for(int g=0;g<nVirt[r];g++)
		{
		  for(int h=0;h<nVirt[s];h++)
		  {
		    i.V2abab_cd((g+v3)*virt+h+v4,(e+v1)*virt+f+v2) = i.V2s[r*nOcc.size()+s][p*nOcc.size()+q]((g+nOcc[r])*nTot[s]+h+nOcc[s],(e+nOcc[p])*nTot[q]+f+nOcc[q]);
		  }
		}
	      }
	    }
	  }
	  else
	  {
	    for(int e=0;e<nVirt[p];e++)
	    {
	      for(int f=0;f<nVirt[q];f++)
	      {
		for(int g=0;g<nVirt[r];g++)
		{
		  for(int h=0;h<nVirt[s];h++)
		  {
		    i.V2abab_cd((g+v3)*virt+h+v4,(e+v1)*virt+f+v2) = 0.0;
		  }
		}
	      }
	    }
	  }
	  v4 += nVirt[s];
	}
	v3 += nVirt[r];
      }
      v2 += nVirt[q];
    }
    v1 += nVirt[p];
  }
  i.V2aaab_cd.set_size(virt*(virt-1)/2,virt*(virt-1)/2);
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      n2=indexD(e+nels,f+nels,nact);
      for(int g=0;g<virt;g++)
      {
	for(int h=g+1;h<virt;h++)
	{
	  i.V2aaab_cd(indexD(g,h,virt),n1) = 2.0*(i.V2abab_cd(g*virt+h,e*virt+f)-i.V2abab_cd(h*virt+g,e*virt+f));
	}
      }
    }
  }
  i.V2abij_ab.set_size(nels*nels,virt*virt);
  v1=0;
  for(int p=0;p<nOcc.size();p++)
  {
    v2=0;
    for(int q=0;q<nOcc.size();q++)
    {
      o3=0;
      for(int r=0;r<nOcc.size();r++)
      {
	o4=0;
	for(int s=0;s<nOcc.size();s++)
	{
	  if(i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_rows>0 && i.V2s[r*nOcc.size()+s][p*nOcc.size()+q].n_cols>0)
	  {
	    for(int e=0;e<nVirt[p];e++)
	    {
	      for(int f=0;f<nVirt[q];f++)
	      {
		for(int m=0;m<nOcc[r];m++)
		{
		  for(int n=0;n<nOcc[s];n++)
		  {
		    i.V2abij_ab((m+o3)*nels+n+o4,(e+v1)*virt+f+v2) = i.V2s[r*nOcc.size()+s][p*nOcc.size()+q](m*nTot[s]+n,(e+nOcc[p])*nTot[q]+f+nOcc[q]);
		  }
		}
	      }
	    }
	  }
	  else
	  {
	    for(int e=0;e<nVirt[p];e++)
	    {
	      for(int f=0;f<nVirt[q];f++)
	      {
		for(int m=0;m<nOcc[r];m++)
		{
		  for(int n=0;n<nOcc[s];n++)
		  {
		    i.V2abij_ab((m+o3)*nels+n+o4,(e+v1)*virt+f+v2) = 0.0;
		  }
		}
	      }
	    }
	  }
	  o4 += nOcc[s];
	}
	o3 += nOcc[r];
      }
      v2 += nVirt[q];
    }
    v1 += nVirt[p];
  }
  i.V2aaij_ab.set_size(nels*(nels-1)/2,virt*(virt-1)/2);
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  i.V2aaij_ab(indexD(m,n,nels),indexD(e,f,virt)) = 2.0*(i.V2abij_ab(m*nels+n,e*virt+f)-i.V2abij_ab(n*nels+m,e*virt+f));
	}
      }
    }
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  for(int t=0;t<100;t++)
  {
    i.ts.push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2-t2));
    i.tLabel.push_back("");
  }

  i.K1s.clear();
  i.V2s.clear();

  return;
}

void readIntsGAMESS(std::string name,int nAO,arma::mat &S,arma::mat &K1)
{
  S.resize(nAO,nAO);
  K1.resize(nAO,nAO);
  S.zeros(); 
  K1.zeros();

  std::ifstream theFile;
  theFile.open(name.c_str());
  std::string str;
  int n; char c;
  double x;

  getline(theFile,str);
  while(!theFile.eof())
  {
    if(str.find("OVERLAP MATRIX") != std::string::npos)
    {
      for(int k=0;k<nAO;k+=5)
      {
	for(int j=k;j<nAO && j<k+5;j++)
	{
	  theFile >> n;
	}
	for(int i=k;i<nAO;i++)
	{
	  theFile >> n; theFile >> str;
	  theFile >> n; theFile >> str;
	  for(int j=k;j<=i && j<k+5;j++)
	  {
	    theFile >> x;
	    S(i,j)=x; S(j,i)=x;
	  }
	}
      }
    }	
    if(str.find("BARE NUCLEUS HAMILTONIAN") != std::string::npos)
    {
      for(int k=0;k<nAO;k+=5)
      {
	for(int j=k;j<nAO && j<k+5;j++)
	{
	  theFile >> n;
	}
	for(int i=k;i<nAO;i++)
	{
	  theFile >> n; theFile >> str;
	  theFile >> n; theFile >> str;
	  for(int j=k;j<=i && j<k+5;j++)
	  {
	    theFile >> x;
	    K1(i,j)=x; K1(j,i)=x;
	  }
	}
      }
    }	
    getline(theFile,str);
  }

  return;
}

void readInts2GAMESS(std::string name,int nAO,arma::mat &V2ab,arma::mat &V2aa)
{
  V2ab.resize(nAO*nAO,nAO*nAO);
  V2aa.resize(nAO*(nAO-1)/2,nAO*(nAO-1)/2);
  V2ab.zeros(); 
  V2aa.zeros();

  std::stringstream oss;
  std::ifstream theFile;
  theFile.open(name.c_str());
  std::string str;
  int n; char c;
  double x;
  int i,j,k,l;
  int nact=nAO;

  getline(theFile,str);
  while(!theFile.eof())
  {
    if(str.find("STORING") != std::string::npos)
    {
      getline(theFile,str);
      while( str.find("TOTAL") == std::string::npos)
      {
	oss.clear();
	oss.str(str);

	for(int a=0; a < str.size(); a+=38)
	{
	  oss >> i >> k >> j >> l >> x;
	  oss >> x;
	  if(i<=nact && j<=nact && k<=nact && l<=nact)
	  {
	    V2ab((i-1)*nact+(j-1),(k-1)*nact+(l-1) )  = x;
	    V2ab((j-1)*nact+(i-1),(l-1)*nact+(k-1) )  = x;
	    V2ab((k-1)*nact+(l-1),(i-1)*nact+(j-1) )  = x;
	    V2ab((l-1)*nact+(k-1),(j-1)*nact+(i-1) )  = x;
	    V2ab((k-1)*nact+(j-1),(i-1)*nact+(l-1) )  = x;
	    V2ab((l-1)*nact+(i-1),(j-1)*nact+(k-1) )  = x;
	    V2ab((i-1)*nact+(l-1),(k-1)*nact+(j-1) )  = x;
	    V2ab((j-1)*nact+(k-1),(l-1)*nact+(i-1) )  = x;
	  }
	}
	getline(theFile,str);
      }
    }	
    getline(theFile,str);
  }

  for(int i=0;i<nact-1;i++)
  {
    for(int j=i+1;j<nact;j++)
    {
      for(int k=0;k<nact-1;k++)
      {
	for(int l=k+1;l<nact;l++)
	{
	  V2aa(indexD(i,j,nact),indexD(k,l,nact))+=2.0*V2ab(index(i,j,nact),index(k,l,nact));
	  V2aa(indexD(i,j,nact),indexD(k,l,nact))-=2.0*V2ab(index(j,i,nact),index(k,l,nact));
	}
      }
    }
  }

  theFile.close();
  return;
}

void genInts(double &Enuc,const std::vector<libint2::Atom> &atoms,std::string basis,std::vector<arma::mat> &Ss,std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V,const std::vector<std::vector<std::vector<orbPair> > > &ao2mo,const std::vector<std::vector<std::vector<int> > > &bas,std::vector<int> nAOs,int &nels,arma::Mat<int> prodTable)
{
  double r1;
  Enuc=0.0;
  nels=0;
  for(int i=0;i<atoms.size();i++)
  {
    nels += atoms[i].atomic_number;
    for(int j=i+1;j<atoms.size();j++)
    {
      r1=(atoms[i].x-atoms[j].x)*(atoms[i].x-atoms[j].x);
      r1+=(atoms[i].y-atoms[j].y)*(atoms[i].y-atoms[j].y);
      r1+=(atoms[i].z-atoms[j].z)*(atoms[i].z-atoms[j].z);
      Enuc += (double) atoms[i].atomic_number*atoms[j].atomic_number/sqrt(r1);
    }
  }

  libint2::BasisSet obs(basis,atoms);

  libint2::init();

  libint2::OneBodyEngine engineS(libint2::OneBodyEngine::overlap,obs.max_nprim(),obs.max_l(),0);
  libint2::OneBodyEngine engineK(libint2::OneBodyEngine::kinetic,obs.max_nprim(),obs.max_l(),0);
  libint2::OneBodyEngine engineN(libint2::OneBodyEngine::nuclear,obs.max_nprim(),obs.max_l(),0);
  std::vector<std::pair<double,std::array<double,3>>> q1;
  for(int a=0;a<atoms.size();a++) 
  {
    q1.push_back( {static_cast<double>(atoms[a].atomic_number), {{atoms[a].x, atoms[a].y, atoms[a].z}}} );
  }
  engineN.set_params(q1);

  auto shell2bf = obs.shell2bf();

  int num1,num2,num3,num4;
  double w1,w2,w3,w4;

  double deg;

  Ss.resize(nAOs.size());
  K1s.resize(nAOs.size());
  for(int r=0;r<nAOs.size();r++)
  {
    Ss[r].resize(nAOs[r],nAOs[r]);
    Ss[r].zeros();
    K1s[r].resize(nAOs[r],nAOs[r]);
    K1s[r].zeros();
  }
  for(int s1=0;s1<obs.size();s1++)
  {
    for(int s2=0;s2<=s1;s2++)
    {
      const auto* intsS = engineS.compute(obs[s1],obs[s2]);
      const auto* intsK = engineK.compute(obs[s1],obs[s2]);
      const auto* intsN = engineN.compute(obs[s1],obs[s2]);

      int bf1 = shell2bf[s1];
      int n1 = obs[s1].size();
      int bf2 = shell2bf[s2];
      int n2 = obs[s2].size();

      deg=1.0;
      if(s1==s2)
      {
	deg=0.5;
      }

      for(int f1=0;f1<n1;f1++)
      {
	for(int r=0;r<Ss.size();r++)
	{
	  for(int i=0;i<ao2mo[bf1+f1][r].size();i++)
	  {
	    num1=ao2mo[bf1+f1][r][i].orb;
	    w1=ao2mo[bf1+f1][r][i].weight;
	    for(int f2=0;f2<n2;f2++)
	    {
	      for(int j=0;j<ao2mo[bf2+f2][r].size();j++)
	      {
		num2=ao2mo[bf2+f2][r][j].orb;
		w2=ao2mo[bf2+f2][r][j].weight;
		Ss[r](num1,num2) += deg*w1*w2*intsS[f1*n2+f2];
		Ss[r](num2,num1) += deg*w1*w2*intsS[f1*n2+f2];
		K1s[r](num1,num2) += deg*w1*w2*intsK[f1*n2+f2];
		K1s[r](num2,num1) += deg*w1*w2*intsK[f1*n2+f2];
		K1s[r](num1,num2) += deg*w1*w2*intsN[f1*n2+f2];
		K1s[r](num2,num1) += deg*w1*w2*intsN[f1*n2+f2];
	      }
	    }
	  }
	}
      }
    }
  }

  libint2::TwoBodyEngine<libint2::Coulomb> engineV(obs.max_nprim(),obs.max_l(), 0);

  int t,nRow,nCol,m;
  double x,y;
  int p,q,r,s;


  V.resize(bas.size()*bas.size(),std::vector<arma::mat>(bas.size()*bas.size(),arma::mat(0,0)));
  for(int t=0;t<bas.size();t++)
  {
    for(int m=0;m<bas[t].size();m++)
    {
      p=bas[t][m][0];
      q=bas[t][m][1];
      for(int n=0;n<bas[t].size();n++)
      {
	r=bas[t][n][0];
	s=bas[t][n][1];
	V[p*bas.size()+q][r*bas.size()+s].set_size(Ss[p].n_rows*Ss[q].n_rows,Ss[r].n_rows*Ss[s].n_rows);
	V[p*bas.size()+q][r*bas.size()+s].zeros();
      }
    }
  }
  for(int I=0;I<obs.size();I++)
  {
    int bf1=shell2bf[I];
    int n1=obs[I].size();
    for(int K=0;K<=I;K++)
    {
      int bf3=shell2bf[K];
      int n3=obs[K].size();
      for(int J=0;J<obs.size();J++)
      {
	int bf2=shell2bf[J];
	int n2=obs[J].size();
	for(int L=0;L<=J && J*obs.size()+L<=I*obs.size()+K;L++)
	{
	  int bf4=shell2bf[L];
	  int n4=obs[L].size();
	  
	  const auto* ints_shellset = engineV.compute(obs[I],obs[K],obs[J],obs[L]);
	  deg=1.0;
	  if(I==K)
	  {
	    deg *= 0.5;
	  }
	  if(J==L)
	  {
	    deg *= 0.5;
	  }
	  if(I==J && K==L)
	  {
	    deg *= 0.5;
	  }
	  for(int f1=0;f1<n1;f1++)
	  {
	    for(int f3=0;f3<n3;f3++)
	    {
	      for(int f2=0;f2<n2;f2++)
	      {
		for(int f4=0;f4<n4;f4++)
		{
		  x = deg*ints_shellset[f1*n3*n2*n4+f3*n2*n4+f2*n4+f4];
		  for(int p=0;p<ao2mo[bf1+f1].size();p++)
		  {
		    for(int i=0;i<ao2mo[bf1+f1][p].size();i++)
		    {
		      num1=ao2mo[bf1+f1][p][i].orb;
		      w1=ao2mo[bf1+f1][p][i].weight;
		      for(int r=0;r<ao2mo[bf3+f3].size();r++)
		      {
			for(int k=0;k<ao2mo[bf3+f3][r].size();k++)
			{
			  num3=ao2mo[bf3+f3][r][k].orb;
			  w3=ao2mo[bf3+f3][r][k].weight;
			  for(int q=0;q<ao2mo[bf2+f2].size();q++)
			  {
			    for(int j=0;j<ao2mo[bf2+f2][q].size();j++)
			    {
			      num2=ao2mo[bf2+f2][q][j].orb;
			      w2=ao2mo[bf2+f2][q][j].weight;
			      
			      t=prodTable(p,q);
			      m=0;
			      while(bas[t][m][0] != r)
			      {
				m++;
			      }
			      s=bas[t][m][1];
			      for(int l=0;l<ao2mo[bf4+f4][s].size();l++)
			      {
				num4=ao2mo[bf4+f4][s][l].orb;
				w4=ao2mo[bf4+f4][s][l].weight;
				V[p*bas.size()+q][r*bas.size()+s](num1*Ss[q].n_rows+num2,num3*Ss[s].n_rows+num4) += w1*w2*w3*w4*x;
				V[q*bas.size()+p][s*bas.size()+r](num2*Ss[p].n_rows+num1,num4*Ss[r].n_rows+num3) += w1*w2*w3*w4*x;
				V[r*bas.size()+s][p*bas.size()+q](num3*Ss[s].n_rows+num4,num1*Ss[q].n_rows+num2) += w1*w2*w3*w4*x;
				V[s*bas.size()+r][q*bas.size()+p](num4*Ss[r].n_rows+num3,num2*Ss[p].n_rows+num1) += w1*w2*w3*w4*x;
			      }
			      t=prodTable(r,q);
			      m=0;
			      while(bas[t][m][0] != p)
			      {
				m++;
			      }
			      s=bas[t][m][1];
			      for(int l=0;l<ao2mo[bf4+f4][s].size();l++)
			      {
				num4=ao2mo[bf4+f4][s][l].orb;
				w4=ao2mo[bf4+f4][s][l].weight;
				V[p*bas.size()+s][r*bas.size()+q](num1*Ss[s].n_rows+num4,num3*Ss[q].n_rows+num2) += w1*w2*w3*w4*x;
				V[s*bas.size()+p][q*bas.size()+r](num4*Ss[p].n_rows+num1,num2*Ss[r].n_rows+num3) += w1*w2*w3*w4*x;
				V[r*bas.size()+q][p*bas.size()+s](num3*Ss[q].n_rows+num2,num1*Ss[s].n_rows+num4) += w1*w2*w3*w4*x;
				V[q*bas.size()+r][s*bas.size()+p](num2*Ss[r].n_rows+num3,num4*Ss[p].n_rows+num1) += w1*w2*w3*w4*x;
			      }
			    }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

  return;
}

void runPara(const std::vector<libint2::Atom> &atoms,std::string basis,const std::vector<int> &nAOs,const std::vector<std::vector<std::vector<orbPair> > > &ao2mo,const std::vector<std::vector<std::vector<int> > > &bas,const arma::Mat<int> &prodTable,double &E,arma::mat &T1,arma::mat &T2aa,arma::mat &T2ab,arma::mat &K1,arma::mat &V2aa,arma::mat &V2ab,bool needT,bool print)
{
  double Enuc,Ecore;
  int nels,nCore;

  std::chrono::high_resolution_clock::time_point t1,t2,t3;
  std::chrono::duration<double> ts;

  int nAO=0;
  std::vector<arma::mat> Cs,Ss,K1s;
  std::vector<std::vector<arma::mat> > V2s;

  setTime(t1,t1,ts);
  setTime(t3,t3,ts);
  if(print)
  {
    std::cout << "Generating integrals" << std::endl;
  }
  genInts(Enuc,atoms,basis,Ss,K1s,V2s,ao2mo,bas,nAOs,nels,prodTable);

  setTime(t1,t2,ts);
  if(print)
  {
    std::cout << "integrals generated" << std::endl;
    std::cout << "time: " << ts.count() << std::endl;
    std::cout << std::endl;
    std::cout << "Enuc: " << Enuc << std::endl;
    std::cout << "nels: " << nels << std::endl;
    nAO=ao2mo.size();
    std::cout << "nAO: " << nAO << std::endl;
  }
  setTime(t1,t1,ts);
  std::vector<int> nOcc(nAOs.size());
  Cs.resize(nAOs.size());
  double Ehf=solveHFsym(nels,Cs,Ss,K1s,V2s,bas,nOcc,nAOs,print);
  if(print)
  {
    std::cout << "Ehf: " << Enuc+Ehf << std::endl;
    setTime(t1,t2,ts);
    std::cout << "HF time: " << ts.count() << std::endl;
  }

  //arma::mat K1,V2aa,V2ab;

  setTime(t1,t2,ts);
  if(print)
  {
    std::cout << "\nrotating integrals" << std::endl;
  }
  rotateInts(K1s,V2s,Cs,bas,K1,V2ab,V2aa,nOcc);
  setTime(t1,t2,ts);
  if(print)
  {
    std::cout << "rotation time: " << ts.count() << std::endl << std::endl;;
  }

  intBundle integrals={0.0,Enuc,K1s,V2s,nOcc,true,nels/2};
  setMats(integrals);

  nels=integrals.nels;
  int virt=integrals.virt;
  int numTot=nels*virt;
  numTot += nels*(nels-1)/2*virt*(virt-1)/2;
  numTot += nels*nels*virt*virt;
  double* Ts;
  Ts = new double[numTot];
  for(int i=0;i<numTot;i++)
  {
    Ts[i]=0.0;
  }

  real_1d_array T;
  T.setcontent(numTot,Ts);

  double epsg = 0.00005;
  double epsf = 0;
  double epsx = 0;
  ae_int_t maxits = 0;
  minlbfgsstate state;
  minlbfgsreport rep;

  minlbfgscreate(3,T,state);
  minlbfgssetcond(state,epsg,epsf,epsx,maxits);
  minlbfgssetxrep(state,true);

  
  setTime(t1,t1,ts);
  if(print)
  {
    std::cout << "Parametric 2-RDM calculation" << std::endl;
    std::cout << "Iter\tE Corr\t\tdE\t\t\tlargest dE/dT" << std::endl;
  }
  if(print)
  {
    minlbfgsoptimize(state,evalE,printIts,&integrals);
  }
  else
  {
    minlbfgsoptimize(state,evalE,NULL,&integrals);
  }
  minlbfgsresults(state,T,rep);
  setTime(t1,t2,ts);

  if(print)
  {
    std::cout << "Total Energy:\t";
    std::cout << (integrals).Ehf+(integrals.E) << std::endl;
    //std::cout << (integrals).Ehf+(integrals.E)+integrals.Enuc << std::endl;
    std::cout << "P2RDM Time: " << ts.count() << std::endl;
  }

  E=integrals.Ehf+integrals.E+integrals.Enuc;

  if(needT)
  {
    T1.set_size(nels,virt);
    int n1a=nels*virt;
    T2aa.set_size(nels*(nels-1)/2,virt*(virt-1)/2);
    int n2aa=T2aa.n_rows*T2aa.n_cols;
    T2ab.set_size(nels*nels,virt*virt);
    for(int e=0;e<virt;e++)
    {
      for(int m=0;m<nels;m++)
      {
	T1(m,e)=T[e*nels+m];
      }
    }
    for(int a=0;a<virt;a++)
    {
      for(int b=a+1;b<virt;b++)
      {
	for(int i=0;i<nels;i++)
	{
	  for(int j=i+1;j<nels;j++)
	  {
	    T2aa(indexD(i,j,nels),indexD(a,b,virt))=T[n1a+indexD(a,b,virt)*T2aa.n_rows+indexD(i,j,nels)];
	  }
	}
      }
    }
    for(int a=0;a<virt;a++)
    {
      for(int b=0;b<virt;b++)
      {
	for(int i=0;i<nels;i++)
	{
	  for(int j=0;j<nels;j++)
	  {
	    T2ab(i*nels+j,a*virt+b)=T[n1a+n2aa+(a*virt+b)*T2ab.n_rows+(i*nels+j)];
	  }
	}
      }
    }
  }

  setTime(t3,t2,ts);
  if(print)
  {
    std::cout << "Total run time: " << ts.count() << std::endl;
  }

  return;
}

void makeD2(const arma::mat &T1,const arma::mat &T2aa,const arma::mat &T2ab,arma::mat &D1,arma::mat &D2aa,arma::mat &D2ab,intBundle2 &integrals)
{
  int nels=T1.n_rows;
  int virt=T1.n_cols;
  int nact=nels+virt;

  int n1,n2,n3,n4,n5,n6,n7,n8;
  double x;

  D1.set_size(nact,nact); D1.zeros();
  D2aa.set_size(nact*(nact-1)/2,nact*(nact-1)/2);  D2aa.zeros();
  D2ab.set_size(nact*nact,nact*nact); D2ab.zeros();

  arma::mat F1m(T1.n_rows,T1.n_cols);
  arma::mat F2aaM(T2aa.n_rows,T2aa.n_cols);
  arma::mat F2abM(T2ab.n_rows,T2ab.n_cols);

  using namespace std::chrono;
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();

  intBundle2 *iPtr=&integrals;
  time_t startTime,lastTime;
  startTime = time(NULL);
  double y;
  int n1a=nels*virt;
  int n2aa=T2aa.n_rows*T2aa.n_cols;
  arma::mat U,Y;

  (*iPtr).E=(*iPtr).Ehf;
  int nT=0;

  std::vector<double> grad(10*(n1+n2aa+T2ab.n_rows*T2ab.n_cols));

  std::chrono::high_resolution_clock::time_point t2;
  std::chrono::duration<double> time_span;
  for(int b=0;b<virt;b++)
  {
    for(int j=0;j<nels;j++)
    {
      //F1m(j,b)=1.0;
      F1m(j,b)=1.0+T1(j,b)*T1(j,b);
    }
  }
  n1=0;
  for(int a=0;a<virt;a++)
  {
    for(int b=0;b<virt;b++)
    {
      n1=a*virt+b;
      n2=0;
      for(int i=0;i<nels;i++)
      {
	for(int j=0;j<nels;j++)
	{
	  //F2abM(n2,n1) = 1.0;
	  F2abM(n2,n1) = 1.0-3.0*T2ab(n2,n1)*T2ab(n2,n1);
	  F2abM(n2,n1) += T1(i,a)*T1(i,a) + T1(j,b)*T1(j,b);
	  n2++;
	}
      }
      n1++;
    }
  }
  n1=0;
  for(int a=0;a<virt;a++)
  {
    for(int b=a+1;b<virt;b++)
    {
      n2=0;
      for(int i=0;i<nels;i++)
      {
	for(int j=i+1;j<nels;j++)
	{
	  F2aaM(n2,n1) = 1.0-3.0*T2aa(n2,n1)*T2aa(n2,n1);
	  F2aaM(n2,n1) += T1(i,a)*T1(i,a) + T1(i,b)*T1(i,b) + T1(j,a)*T1(j,a) + T1(j,b)*T1(j,b);
	  n2++;
	}
      }
      n1++;
    }
  }

  /*for(int m=0;m<nels;m++)
  {
    (*iPtr).E += 2.0*(*iPtr).K1i_j(m,m);
  }*/
  n1=0;  
  for(int m=0;m<nels;m++)
  {
    n1=m*nact;
    for(int n=0;n<nels;n++)
    {
      n2=n1+n;
      //D2ab(n2,n2) = 1.0;
      (*iPtr).E += 2.0*(*iPtr).V2abij_kl(m*nels+n,m*nels+n)-(*iPtr).V2abij_kl(n*nels+m,m*nels+n);
    }
  }
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      //D2aa(indexD(m,n,nels),indexD(m,n,nels)) = 1.0;
    }
  }
  //t1
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Wab_ij_kl=T2ab*T2ab.t();
  arma::mat W1_i_j=-T1*T1.t();
  for(int m=0;m<nels;m++)
  {
    //x = -dot(T1.row(m),T1.row(m));
    x = W1_i_j(m,m);
    for(int a=0;a<virt;a++)
    {
      F1m(m,a) += x;
      for(int n=0;n<nels;n++)
      {
	n3=m*nels+n;
	n5=n*nels+m;
	for(int b=0;b<virt;b++)
	{
	  n4=a*virt+b;
	  F2abM(n3,n4) += x;
	  F2abM(n5,n4) += x;
	  //F2abM(m*nels+n,a*virt+b) += x;
	  //F2abM(n*nels+m,a*virt+b) += x;
	}
      }
      for(int b=a+1;b<virt;b++)
      {
	n2 = indexD(a,b,virt);
	for(int n=0;n<m;n++)
	{
	  n1 = indexD(n,m,nels);
	  F2aaM(n1,n2) += x;
	  //F2aaM(indexD(n,m,nels),indexD(a,b,virt)) += x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  n1= indexD(m,n,nels);
	  F2aaM(n1,n2) += x;
	  //F2aaM(indexD(m,n,nels),indexD(a,b,virt)) += x;
	}
      }
    }
  }
  //t2
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int m=0;m<nels;m++)
  {
    for(int n=0;n<nels;n++)
    {
      n1 = m*nels+n;
      x = Wab_ij_kl(n1,n1);
      //x = dot(T2ab.row(n1),T2ab.row(n1));
      //x = W(m*nels+n,m*nels+n);
      for(int e=0;e<virt;e++)
      {
	for(int f=0;f<virt;f++)
	{
	  n2 = e*virt+f;
	  F2abM(n1,n2) -= x;
	  //F2abM(m*nels+n,e*virt+f) -= x;
	}
      }
    }
  }
  //t3

  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int i=0;i<nels;i++)
  {
    for(int j=0;j<nels;j++)
    {
      n1 = i*nact+j;
      n5 = i*nels+j;
      for(int k=0;k<nels;k++)
      {
	n2 = i*nels+k;
	n3 = j*nels+k;
	W1_i_j(i,j) -= Wab_ij_kl(n2,n3);
	//V(i,j) -= W(i*nels+k,j*nels+k);
	for(int l=0;l<nels;l++)
	{
	  n4 = k*nact+l;
	  n6 = k*nels+l;
	  D2ab(n1,n4)+=Wab_ij_kl(n5,n6);
	  //D2ab(i*nact+j,k*nact+l)+=W(i*nels+j,k*nels+l);
	}
      }
    }
  }
  //t4
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Waa_ij_kl=T2aa*T2aa.t();
  (*iPtr).E += trace((*iPtr).V2aaij_kl*T2aa*T2aa.t());
  for(int i=0;i<nels;i++)
  {
    for(int j=0;j<nels;j++)
    {
      for(int k=0;k<std::min(i,j);k++)
      {
	n1 = indexD(k,i,nels);
	n2 = indexD(k,j,nels);
	W1_i_j(i,j) -= Waa_ij_kl(n1,n2);
	//V(i,j) -= W(indexD(k,i,nels),indexD(k,j,nels));
      }
      for(int k=i+1;k<j;k++)
      {
	n1 = indexD(i,k,nels);
	n2 = indexD(k,j,nels);
	W1_i_j(i,j) += Waa_ij_kl(n1,n2);
	//V(i,j) += W(indexD(i,k,nels),indexD(k,j,nels));
      }
      for(int k=j+1;k<i;k++)
      {
	n1 = indexD(k,i,nels);
	n2 = indexD(j,k,nels);
	W1_i_j(i,j) += Waa_ij_kl(n1,n2);
	//V(i,j) += W(indexD(k,i,nels),indexD(j,k,nels));
      }
      for(int k=std::max(i,j)+1;k<nels;k++)
      {
	n1 = indexD(i,k,nels);
	n2 = indexD(j,k,nels);
	W1_i_j(i,j) -= Waa_ij_kl(n1,n2);
	//V(i,j) -= W(indexD(i,k,nels),indexD(j,k,nels));
      }
      for(int k=i+1;k<nels;k++)
      {
	n3=indexD(i,k,nels);
	n1 = indexD(i,k,nact);
	for(int l=j+1;l<nels;l++)
	{
	  n4=indexD(j,l,nels);
	  n2=indexD(j,l,nact);
	  D2aa(n1,n2)+=Waa_ij_kl(n3,n4);
	  //D2aa(n1,n2)+=W(indexD(i,k,nels),indexD(j,l,nels));
	}
      }
    }
  }
  //t5
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  D1.submat(0,0,nels-1,nels-1)=W1_i_j;
  //(*iPtr).E += 2.0*accu((*iPtr).V1i_j%W1_i_j);
  //(*iPtr).E += 2.0*trace((*iPtr).V1i_j*W1_i_j);
  for(int i=0;i<nels;i++)
  {
    //D1(i,i) += 1.0;
    for(int j=0;j<nels;j++)
    {
      n1 = i*nact+j;
      n2 = j*nact+i;
      for(int k=0;k<nels;k++)
      {
	n3 = i*nact+k;
	n4 = k*nact+i;
	D2ab(n1,n3) += W1_i_j(j,k);
	D2ab(n2,n4) += W1_i_j(j,k);
	//D2ab(i*nact+j,i*nact+k) += V(j,k);
	//D2ab(j*nact+i,k*nact+i) += V(j,k);
      }
    }
    for(int j=i+1;j<nels;j++)
    {
      n1 = indexD(i,j,nact);
      for(int k=0;k<i;k++)
      {
	n2 = indexD(k,i,nact);
	D2aa(n1,n2)-=W1_i_j(j,k);
      }
      for(int k=i+1;k<nels;k++)
      {
	n2 = indexD(i,k,nact);
	D2aa(n1,n2)+=W1_i_j(j,k);
      }
      for(int k=0;k<j;k++)
      {
	n2 = indexD(k,j,nact);
	D2aa(n1,n2)+=W1_i_j(i,k);
      }
      for(int k=j+1;k<nels;k++)
      {
	n2 = indexD(j,k,nact);
	D2aa(n1,n2)-=W1_i_j(i,k);
      }
    }
  }
  //t6
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      n1 = indexD(m,n,nels);
      //x = dot(T2aa.row(n1),T2aa.row(n1));
      x = Waa_ij_kl(n1,n1);
      //x = W(indexD(m,n,nels),indexD(m,n,nels));
      for(int e=0;e<virt;e++)
      {
	for(int f=e+1;f<virt;f++)
	{
	  n2=indexD(e,f,virt);
	  F2aaM(n1,n2) -= x;
	  //F2aaM(indexD(m,n,nels),indexD(e,f,virt)) -= x;
	}
      }
    }
  }
  //t7
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Wab_ab_cd=T2ab.t()*T2ab;
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  arma::mat W1_a_b=T1.t()*T1;
  for(int a=0;a<virt;a++)
  {
    //x = dot(T1.col(a),T1.col(a));
    x = W1_a_b(a,a);
    for(int m=0;m<nels;m++)
    {
      F1m(m,a) -= x;
      for(int n=0;n<nels;n++)
      {
	n1=m*nels+n;
	for(int b=0;b<virt;b++)
	{
	  n2=a*virt+b;
	  n3=b*virt+a;
	  F2abM(n1,n2) -= x;
	  F2abM(n1,n3) -= x;
	  //F2abM(m*nels+n,a*virt+b) -= x;
	  //F2abM(m*nels+n,b*virt+a) -= x;
	}
      }
      for(int n=m+1;n<nels;n++)
      {
	n1 = indexD(m,n,nels);
	for(int b=0;b<a;b++)
	{
	  n2=indexD(b,a,virt);
	  F2aaM(n1,n2) -= x;
	  //F2aaM(indexD(m,n,nels),indexD(b,a,virt)) -= x;
	}
	for(int b=a+1;b<virt;b++)
	{
	  n2=indexD(a,b,virt);
	  F2aaM(n1,n2) -= x;
	  //F2aaM(indexD(m,n,nels),indexD(a,b,virt)) -= x;
	}
      }
    }
  }
  //t8
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {	
      n1=(e+nels)*nact+f+nels;
      n2=e*virt+f;
      x=0.0;
      for(int g=0;g<virt;g++)
      {
	n3=e*virt+g;
	n4=f*virt+g;
	x += Wab_ab_cd(n3,n4);
	//V(e,f) += W(e*virt+g,f*virt+g);
	for(int h=0;h<virt;h++)
	{
	  n5=(g+nels)*nact+h+nels;
	  n6=g*virt+h;
	  D2ab(n1,n5) += Wab_ab_cd(n2,n6);
	  //D2ab((e+nels)*nact+f+nels,(g+nels)*nact+h+nels) += W(e*virt+f,g*virt+h);
	}
      }
      W1_a_b(e,f) += x;
      x = Wab_ab_cd(n2,n2);
      //x = dot(T2ab.col(n2),T2ab.col(n2));
      //x = W(e*virt+f,e*virt+f);
      for(int m=0;m<nels;m++)
      {
	for(int n=0;n<nels;n++)
	{
	  n7=m*nels+n;
	  F2abM(n7,n2) -= x;
	  //F2abM(m*nels+n,e*virt+f) -= x;
	}
      }
    }
  }
  //t9
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Waa_ab_cd=T2aa.t()*T2aa;
  (*iPtr).E += trace((*iPtr).V2aaab_cd*T2aa.t()*T2aa);
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {
      for(int g=0;g<std::min(e,f);g++)
      {
	W1_a_b(e,f) += Waa_ab_cd(indexD(g,e,virt),indexD(g,f,virt));
      }
      for(int g=e+1;g<f;g++)
      {
	W1_a_b(e,f) -= Waa_ab_cd(indexD(e,g,virt),indexD(g,f,virt));
      }
      for(int g=f+1;g<e;g++)
      {
	W1_a_b(e,f) -= Waa_ab_cd(indexD(g,e,virt),indexD(f,g,virt));
      }
      for(int g=std::max(e,f)+1;g<virt;g++)
      {
	W1_a_b(e,f) += Waa_ab_cd(indexD(e,g,virt),indexD(f,g,virt));
      }
      for(int g=e+1;g<virt;g++)
      {
	n1 = indexD(e+nels,g+nels,nact);
	n3 = indexD(e,g,virt);
	for(int h=f+1;h<virt;h++)
	{
	  n2 = indexD(f+nels,h+nels,nact);
	  n4 = indexD(f,h,virt);
	  D2aa(n1,n2) += Waa_ab_cd(n3,n4);
	  //D2aa(n1,n2) += W(indexD(e,g,virt),indexD(f,h,virt));
	}
      }
    }
  }
  std::cout << "hello" << std::endl;
  //t10
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  D1.submat(nels,nels,nact-1,nact-1)=W1_a_b;
  //(*iPtr).E += 2.0*accu((*iPtr).V1a_b%W1_a_b);
  //(*iPtr).E += 2.0*trace((*iPtr).V1a_b*W1_a_b);
  for(int m=0;m<nels;m++)
  {
    for(int e=0;e<virt;e++)
    {
      n3=m*nact+e+nels;;
      n4=(e+nels)*nact+m;
      n1 = indexD(m,e+nels,nact);
      for(int f=0;f<virt;f++)
      {
	n5=m*nact+f+nels;;
	n6=(f+nels)*nact+m;
	D2ab(n3,n5) += W1_a_b(e,f);
	D2ab(n4,n6) += W1_a_b(e,f);
	//D2ab(m*nact+e+nels,m*nact+f+nels) += V(e,f);
	//D2ab((e+nels)*nact+m,(f+nels)*nact+m) += V(e,f);
	D2aa(n1,indexD(m,f+nels,nact)) += W1_a_b(e,f);
      }
    }
  }
  //t11

  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1 = indexD(e,f,virt);
      //x = dot(T2aa.col(n1),T2aa.col(n1));
      x = Waa_ab_cd(n1,n1);
      //x = W(indexD(e,f,virt),indexD(e,f,virt));
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n1) -= x;
	  //F2aaM(indexD(m,n,nels),indexD(e,f,virt)) -= x;
	}
      }
    }
  }
  //t12
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Wab_ia_jb2(nels*virt,nels*virt);
  for(int a=0;a<virt;a++)
  {
    for(int b=0;b<virt;b++)
    {
      n3=a*virt+b;
      for(int i=0;i<nels;i++)
      {
	n1=a*nels+i;
	for(int j=0;j<nels;j++)
	{
	  Wab_ia_jb2(b*nels+j,n1) = T2ab(j*nels+i,n3);
	  //W(j*virt+b,i*virt+a) = T2ab(j*nels+i,a*virt+b);
	}
      }
    }
  }
  //t13
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  (*iPtr).E -= 2.0*trace((*iPtr).V2abia_jb2*Wab_ia_jb2*Wab_ia_jb2.t());
  //(*iPtr).E -= 2.0*trace((*iPtr).V2abia_jb2*Wab_ia_jb2*Wab_ia_jb2.t());
  arma::mat Vab_ia_jb2=Wab_ia_jb2*Wab_ia_jb2.t();
  for(int f=0;f<virt;f++)
  {
    for(int m=0;m<nels;m++)
    {
      n1=f*nels+m;
      for(int n=0;n<nels;n++)
      {
	n2=(f+nels)*nact+n;
	n3=n*nact+f+nels;
	for(int e=0;e<virt;e++)
	{
	  D2ab(m*nact+e+nels,n3) -= Vab_ia_jb2(e*nels+n,n1);
	  D2ab((e+nels)*nact+m,n2) -= Vab_ia_jb2(e*nels+n,n1);
	  //D2ab(m*nact+e+nels,n*nact+f+nels)-=U(n*virt+e,m*virt+f);
	  //D2ab((e+nels)*nact+m,(f+nels)*nact+n)-=U(n*virt+e,m*virt+f);
	}
      }
    }
  }

  //t14
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  //arma::mat Vab_ia_jb2 = Wab_ia_jb2*Wab_ia_jb2.t();
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n1=e*nels+m;
      //x = dot(Wab_ia_jb2.row(n1),Wab_ia_jb2.row(n1));
      x = Vab_ia_jb2(n1,n1);
      //x = U(m*virt+e,m*virt+e);
      for(int f=0;f<virt;f++)
      {
	n2=e*virt+f;
	n3=f*virt+e;
	for(int n=0;n<nels;n++)
	{
	  F2abM(m*nels+n,n3) -= x;
	  F2abM(n*nels+m,n2) -= x;
	  //F2abM(m*nels+n,f*virt+e) -= x;
	  //F2abM(n*nels+m,e*virt+f) -= x;
	}
      }
    }
  }
  //t15
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  U=(*iPtr).V2abia_jb2*Wab_ia_jb2;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels;n++)
	{
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+(m*nels+n)] -= 2.0*(U(f*nels+m,e*nels+n)+U(e*nels+n,f*nels+m)); 
	}
      }
    }
  }
  //t16
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Wab_ia_jb(nels*virt,nels*virt);
  for(int a=0;a<virt;a++)
  {
    for(int i=0;i<nels;i++)
    {
      n1=a*nels+i;
      for(int b=0;b<virt;b++)
      {
	n2=a*virt+b;
	for(int j=0;j<nels;j++)
	{
	  Wab_ia_jb(b*nels+j,n1) = T2ab(i*nels+j,n2);
	  //W(j*virt+b,i*virt+a) = T2ab(i*nels+j,a*virt+b);
	}
      }
    }
  }
  //t17
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  arma::mat Vab_ia_jb=Wab_ia_jb*Wab_ia_jb.t();
  (*iPtr).E -= 2.0*trace((*iPtr).V2aaia_jb*Wab_ia_jb*Wab_ia_jb.t());
  arma::mat W1aa_ia_jb(nels*virt,nels*virt);
  for(int a=0;a<virt;a++)
  {
    for(int i=0;i<nels;i++)
    {
      for(int b=0;b<virt;b++)
      {
	for(int j=0;j<nels;j++)
	{
	  W1aa_ia_jb(b*nels+j,a*nels+i) = T1(i,a)*T1(j,b);
	}
      }
    }
  }
  //(*iPtr).E -= 2.0*trace((*iPtr).V2aaia_jb*W1aa_ia_jb);
  for(int f=0;f<virt;f++)
  {
    for(int m=0;m<nels;m++)
    {
      n3=f*nels+m;
      //F1m(m,f) += Vab_ia_jb(n3,n3);
      //F1m(m,f) += U(m*virt+f,m*virt+f);
      for(int e=0;e<virt;e++)
      {
	n1 = indexD(m,e+nels,nact);
	for(int n=0;n<nels;n++)
	{
	  n2 = indexD(n,f+nels,nact);
	  D2aa(n2,n1) -= T1(n,e)*T1(m,f);
	  D2aa(n2,n1) -= Vab_ia_jb(e*nels+n,n3);
	  //D2aa(n1,n2) -= U(n*virt+e,m*virt+f);
	}
      }
    }
  }
  //t18
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n1=e*nels+m;
      x=dot(Wab_ia_jb.col(n1),Wab_ia_jb.col(n1));
      F1m(m,e) += x;
      //x = U(m*virt+e,m*virt+e);
      for(int f=0;f<virt;f++)
      {
	n2=e*virt+f;
	n3=f*virt+e;
	for(int n=0;n<nels;n++)
	{
	  F2abM(n*nels+m,n3) -= x;
	  F2abM(m*nels+n,n2) -= x;
	  //F2abM(n*nels+m,f*virt+e) -= x;
	  //F2abM(m*nels+n,e*virt+f) -= x;
	}
      }
      for(int f=0;f<e;f++)
      {
	n2=indexD(f,e,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n2) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n2) -= x;
	}
      }
      for(int f=e+1;f<virt;f++)
      {
	n2=indexD(e,f,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n2) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n2) -= x;
	}
      }
    }
  }
  //t19
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Waa_ia_jb(nels*virt,nels*virt);
//V_ai_fm = T2aa_mi_fa
  for(int a=0;a<virt;a++)
  {
    for(int i=0;i<nels;i++)
    {
      n3=a*nels+i;
      for(int m=0;m<nels;m++)
      {
	Waa_ia_jb(n3,a*nels+m)=0.0;
	//V(i*virt+a,i*virt+b)=0.0;
      }
      for(int f=0;f<a;f++)
      {
	n1=indexD(f,a,virt);
	Waa_ia_jb(n3,f*nels+i)=0.0;
	for(int m=0;m<i;m++)
	{
	  Waa_ia_jb(n3,f*nels+m) = T2aa(indexD(m,i,nels),n1);
	  //V(i*virt+a,m*virt+f) = T2aa(indexD(m,i,nels),indexD(f,a,virt));
	}
	for(int m=i+1;m<nels;m++)
	{
	  Waa_ia_jb(n3,f*nels+m) = -T2aa(indexD(i,m,nels),n1);
	}
      }
      for(int f=a+1;f<virt;f++)
      {
	n1=indexD(a,f,virt);
	Waa_ia_jb(n3,f*nels+i)=0.0;
	for(int m=0;m<i;m++)
	{
	  Waa_ia_jb(n3,f*nels+m) = -T2aa(indexD(m,i,nels),n1);
	}
	for(int m=i+1;m<nels;m++)
	{
	  Waa_ia_jb(n3,f*nels+m) = T2aa(indexD(i,m,nels),n1);
	}
      }
    }
  }
  //t20
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  arma::mat W1_i_a = (Wab_ia_jb+Waa_ia_jb)*arma::vectorise(T1);

  arma::mat Vaa_ia_jb=Waa_ia_jb.t()*Waa_ia_jb;
  (*iPtr).E -= 2.0*trace((*iPtr).V2aaia_jb*Waa_ia_jb.t()*Waa_ia_jb);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n3=e*nels+m;
      n1=indexD(m,e+nels,nact);
      //F1m(m,e) += Vaa_ia_jb(n3,n3);
      //F1m(m,e) += U(m*virt+e,m*virt+e);
      for(int n=0;n<nels;n++)
      {
	n4=e*nels+n;
	for(int f=0;f<virt;f++)
	{
	  D2aa(indexD(n,f+nels,nact),n1) -= Vaa_ia_jb(f*nels+m,n4);
	  //D2aa(indexD(m,e+nels,nact),indexD(n,f+nels,nact)) -= U(n4,m*virt+f);
	}
      }
    }
  }
  //t21
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n3=e*nels+m;
      x=Vaa_ia_jb(n3,n3);
      //x = dot(Waa_ia_jb.col(n3),Waa_ia_jb.col(n3));
      F1m(m,e) += x;
      //x = U(m*virt+e,m*virt+e);
      for(int f=0;f<virt;f++)
      {
	n4=e*virt+f;
	n5=f*virt+e;
	for(int n=0;n<nels;n++)
	{
	  F2abM(m*nels+n,n4) -= x;
	  F2abM(n*nels+m,n5) -= x;
	  //F2abM(m*nels+n,e*virt+f) -= x;
	  //F2abM(n*nels+m,f*virt+e) -= x;
	}
      }
      for(int f=0;f<e;f++)
      {
	n1 = indexD(f,e,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n1) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n1) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
      }
      for(int f=e+1;f<virt;f++)
      {
	n1 = indexD(e,f,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n1) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n1) -= x;
	  //F2aaM(indexD(n,m,nels),indexD(f,e,virt)) -= x;
	}
      }
    }
  }
  //t22
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  arma::mat Uab_ia_jb = Wab_ia_jb*Waa_ia_jb;
  arma::mat W1ia_jb(nels*virt,nels*virt);
  for(int b=0;b<virt;b++)
  {
    for(int j=0;j<nels;j++)
    {
      for(int a=0;a<virt;a++)
      {
	for(int i=0;i<nels;i++)
	{
	  W1ia_jb(b*nels+j,a*nels+i) = T1(i,a)*T1(j,b);
	}
      }
    }
  }

  (*iPtr).E += 4.0*trace((*iPtr).V2abia_jb*Wab_ia_jb*Waa_ia_jb);
  (*iPtr).E += 2.0*trace((*iPtr).V2abia_jb*W1ia_jb);
  (*iPtr).E -= 2.0*trace((*iPtr).V2aaia_jb*W1ia_jb);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n1=m*nact+e+nels;
      n2=(e+nels)*nact+m;
      for(int f=0;f<virt;f++)
      {
	n3=f*nels+m;
	for(int n=0;n<nels;n++)
	{
	  //x = Uab_ia_jb(e*nels+n,n3) + Uab_ia_jb(n3,e*nels+n);
	  //x = T1(n,e)*T1(m,f);
	  x = T1(n,e)*T1(m,f) + Uab_ia_jb(e*nels+n,n3) + Uab_ia_jb(n3,e*nels+n);
	  //x = T1(n,e)*T1(m,f) + U(n*virt+e,m*virt+f) + U(m*virt+f,n*virt+e);
	  D2ab(n1,(f+nels)*nact+n) += x;
	  D2ab(n2,n*nact+f+nels) += x;
	  //D2ab(m*nact+e+nels,(f+nels)*nact+n) += x;
	  //D2ab((e+nels)*nact+m,n*nact+f+nels) += x;
	}
      }
    }
  }
  //t23
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  U=(*iPtr).K1ia*(Wab_ia_jb+Waa_ia_jb);
  //U=((*iPtr).K1ia+(*iPtr).V1ia+(*iPtr).V1aia)*(Wab_ia_jb+Waa_ia_jb);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m]+=4.0*U(0,e*nels+m);
    }
  }
  //t24
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  U=(*iPtr).V2abia_jb*Waa_ia_jb-(*iPtr).V2aaia_jb*Wab_ia_jb;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+m*nels+m] += 4.0*U(e*nels+m,e*nels+m);
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels && f*nels+n<e*nels+m;n++)
	{
	  x = 2.0*(U(f*nels+n,e*nels+m)+U(e*nels+m,f*nels+n));
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+m*nels+n] += x;
	  grad[n1a+n2aa+(f*virt+e)*T2ab.n_rows+n*nels+m] += x;
	}
      }
    }
  }
  //t25
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  U=(*iPtr).V2abia_jb*Wab_ia_jb-(*iPtr).V2aaia_jb*Waa_ia_jb;
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  x=4.0*(U(e*nels+m,f*nels+n)-U(f*nels+m,e*nels+n)-U(e*nels+n,f*nels+m)+U(f*nels+n,e*nels+m));
	  //E2grad[n1a+n1*T2aa.n_rows+indexD(m,n,nels)] += x;
	  grad[n1a+n1*T2aa.n_rows+indexD(m,n,nels)] += x;
	}
      }
    }
  }
  //t26
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Wab_i_jab(nels,nels*virt*virt);
  for(int f=0;f<virt;f++)
  {
    for(int g=0;g<virt;g++)
    {
      n1=f*virt+g;
      for(int m=0;m<nels;m++)
      {
	n2=n1*nels+m;
	for(int i=0;i<nels;i++)
	{
	  Wab_i_jab(i,n2) = T2ab(m*nels+i,n1);
	  //W(i,(f*virt+g)*nels + m) = T2ab(m*nels+i,f*virt+g);
	}
      }
    }
  }
  //(*iPtr).E -= 2.0*trace((*iPtr).V1i_j*Wab_i_jab*Wab_i_jab.t());
  //(*iPtr).E -= 2.0*trace((*iPtr).V1i_j*T1*T1.t());
  (*iPtr).E -= 2.0*trace((*iPtr).K1i_j*Wab_i_jab*Wab_i_jab.t());
  (*iPtr).E -= 2.0*trace((*iPtr).K1i_j*T1*T1.t());
  //(*iPtr).E -= 2.0*trace((*iPtr).V1ai_j*Wab_i_jab*Wab_i_jab.t());
  //(*iPtr).E -= 2.0*trace((*iPtr).V1ai_j*T1*T1.t());
  
  //t27
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Vab_a_jbc = T1.t()*Wab_i_jab;
  (*iPtr).E += 4.0*trace((*iPtr).V2abh_fgn*Wab_i_jab.t()*T1);
  for(int f=0;f<virt;f++)
  {
    for(int g=0;g<virt;g++)
    {
      n2=(f+nels)*nact+g+nels;
      n3=(g+nels)*nact+f+nels;
      n6=(f*virt+g)*nels;
      for(int m=0;m<nels;m++)
      {
	n1=n6+m;
	for(int e=0;e<virt;e++)
	{
	  n4=m*nact+e+nels;
	  n5=(e+nels)*nact+m;
	  x = Vab_a_jbc(e,n1);
	  D2ab(n4,n2) += x;
	  D2ab(n2,n4) += x;
	  D2ab(n5,n3) += x;
	  D2ab(n3,n5) += x;
	  //x = U(e,(f*virt+g)*nels+m);
	  //D2ab(m*nact+e+nels,(f+nels)*nact+g+nels) += x;
	  //D2ab((f+nels)*nact+g+nels,m*nact+e+nels) += x;
	  //D2ab((e+nels)*nact+m,(g+nels)*nact+f+nels) += x;
	  //D2ab((g+nels)*nact+f+nels,(e+nels)*nact+m) += x;
	}
      }
    }
  }
  //t28
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  U=Wab_i_jab*(*iPtr).V2abh_fgn.t();
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m] += 4.0*U(m,e);
    }
  }
  //t29
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  U=(*iPtr).K1i_j*Wab_i_jab;
  //U=((*iPtr).K1i_j+(*iPtr).V1i_j+(*iPtr).V1ai_j)*Wab_i_jab;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+(m*nels+m)]-=4.0*U(m,(e*virt+e)*nels+m);
      //E2grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+(m*nels+m)]-=4.0*U(m,(e*virt+e)*nels+m);
      //E2agrad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+(m*nels+m)]-=4.0*U(m,(e*virt+e)*nels+m);
      for(int f=0;f<virt;f++)
      {
	n1=e*virt+f;
	n2=n1*nels+m;
	n3=f*virt+e;
	for(int n=0;n<nels && f*nels+n<e*nels+m;n++)
	{
	  x = U(n,n2) + U(m,n3*nels+n);
	  //E2agrad[n1a+n2aa+n1*T2ab.n_rows+m*nels+n] -= 2.0*x;
	  //E2agrad[n1a+n2aa+n3*T2ab.n_rows+n*nels+m] -= 2.0*x;
	  //E2grad[n1a+n2aa+n1*T2ab.n_rows+m*nels+n] -= 2.0*x;
	  //E2grad[n1a+n2aa+n3*T2ab.n_rows+n*nels+m] -= 2.0*x;
	  grad[n1a+n2aa+n1*T2ab.n_rows+m*nels+n] -= 2.0*x;
	  grad[n1a+n2aa+n3*T2ab.n_rows+n*nels+m] -= 2.0*x;
	}
      }
    }
  }
  //t30
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int f=0;f<virt;f++)
  {
    for(int g=0;g<virt;g++)
    {
      n3=f*virt+g;
      n4=g*virt+f;
      for(int m=0;m<nels;m++)
      {
	n1=n3*nels+m;
	Y = Wab_i_jab.col(n1).t()*Wab_i_jab.col(n1);
	//Y = W.col((f*virt+g)*nels+m).t()*W.col((f*virt+g)*nels+m);
	for(int n=0;n<nels;n++)
	{
	  F2abM(m*nels+n,n3) += 2.0*Y(0);
	  F2abM(n*nels+m,n4) += 2.0*Y(0);
	  //F2abM(m*nels+n,f*virt+g) += 2.0*Y(0);
	  //F2abM(n*nels+m,g*virt+f) += 2.0*Y(0);
	}
      }
    }
  }
  //t31
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int m=0;m<nels;m++)
  {
    Y = Wab_i_jab.row(m)*Wab_i_jab.row(m).t();
    for(int e=0;e<virt;e++)
    {
      F1m(m,e) -= Y(0);
    }
  }
  //t32
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Waa_i_jab(nels,nels*virt*(virt-1)/2);
  for(int f=0;f<virt;f++)
  {
    for(int g=f+1;g<virt;g++)
    {
      n2=indexD(f,g,virt);
      for(int m=0;m<nels;m++)
      {
	n3=n2*nels+m;
	Waa_i_jab(m,n3)=0.0;;
	//W(m,indexD(f,g,virt)*nels+m) = 0.0;
	for(int i=0;i<m;i++)
	{
	  Waa_i_jab(i,n3) = -T2aa(indexD(i,m,nels),n2);
	  //W(i,indexD(f,g,virt)*nels+m) = -T2aa(indexD(i,m,nels),indexD(f,g,virt));
	}
	for(int i=m+1;i<nels;i++)
	{
	  Waa_i_jab(i,n3) = T2aa(indexD(m,i,nels),n2);
	  //W(i,indexD(f,g,virt)*nels+m) = T2aa(indexD(m,i,nels),indexD(f,g,virt));
	}
      }
    }
  }
  //(*iPtr).E -= 2.0*trace((*iPtr).V1i_j*Waa_i_jab*Waa_i_jab.t());
  (*iPtr).E -= 2.0*trace((*iPtr).K1i_j*Waa_i_jab*Waa_i_jab.t());
  //(*iPtr).E -= 2.0*trace((*iPtr).V1ai_j*Waa_i_jab*Waa_i_jab.t());
  //t33
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  arma::mat Vaa_a_jbc = T1.t()*Waa_i_jab;
  (*iPtr).E += 2.0*trace((*iPtr).V2aah_fgn*Waa_i_jab.t()*T1);
  for(int f=0;f<virt;f++)
  {
    for(int g=f+1;g<virt;g++)
    {
      n3 = indexD(f,g,virt)*nels;
      n1=indexD(f+nels,g+nels,nact);
      for(int m=0;m<nels;m++)
      {
	n4=n3+m;
	for(int e=0;e<virt;e++)
	{
	  x = Vaa_a_jbc(e,n4);
	  //x = U(e,(indexD(f,g,virt))*nels+m);
      //n1=indexD(f+nels,g+nels,nact);
	  n2=indexD(m,e+nels,nact);
	  D2aa(n1,n2) += x;
	  D2aa(n2,n1) += x;
//	  D2aa(indexD(f+nels,g+nels,nact),indexD(m,e+nels,nact)) += x;
//	  D2aa(indexD(m,e+nels,nact),indexD(f+nels,g+nels,nact)) += x;
	}
      }
    }
  }
  //t34
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  U=(*iPtr).K1i_j*Waa_i_jab;
  //U=((*iPtr).K1i_j+(*iPtr).V1i_j+(*iPtr).V1ai_j)*Waa_i_jab;
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  n2=indexD(m,n,nels);
	  //E2agrad[n1a+n1*T2aa.n_rows+n2] += 4.0*(U(m,n1*nels+n)-U(n,n1*nels+m));
	  //E2grad[n1a+n1*T2aa.n_rows+n2] += 4.0*(U(m,n1*nels+n)-U(n,n1*nels+m));
	  grad[n1a+n1*T2aa.n_rows+n2] += 4.0*(U(m,n1*nels+n)-U(n,n1*nels+m));
	}
      }
    }
  }
  //t35
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  U=(*iPtr).V2aah_fgn*Waa_i_jab.t();
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m] += 2.0*U(e,m);
    }
  }
  //t36
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int m=0;m<nels;m++)
  {
    Y = Waa_i_jab.row(m)*Waa_i_jab.row(m).t();
    for(int e=0;e<virt;e++)
    {
      F1m(m,e) -= Y(0);
    }
  }
  //t37
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n2 = indexD(e,f,virt);
      for(int m=0;m<nels;m++)
      {
	n3=n2*nels+m;
	Y = Waa_i_jab.col(n3).t()*Waa_i_jab.col(n3);
	//Y = W.col(n2*nels+m).t()*W.col(n2*nels+m);
	//Y = W.col(indexD(e,f,virt)*nels+m).t()*W.col(indexD(e,f,virt)*nels+m);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n2) += 2.0*Y(0);
	  //F2aaM(indexD(n,m,nels),indexD(e,f,virt)) += 2.0*Y(0);
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n2) += 2.0*Y(0);
	  //F2aaM(indexD(m,n,nels),indexD(e,f,virt)) += 2.0*Y(0);
	}
      }
    }
  }
  //t38
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Wab_ija_b(nels*nels*virt,virt);
//W_ija_b = T2ab_ij_ba
  for(int i=0;i<nels;i++)
  {
    for(int j=0;j<nels;j++)
    {
      n2=i*nels+j;
      for(int a=0;a<virt;a++)
      {
	n1=n2*virt+a;
	for(int b=0;b<virt;b++)
	{
	  Wab_ija_b(n1,b)=T2ab(n2,b*virt+a);
	  //W((i*nels+j)*virt+b,a)=T2ab(i*nels+j,a*virt+b);
	}
      }
    }
  }
  (*iPtr).E += 2.0*trace((*iPtr).V1a_b*Wab_ija_b.t()*Wab_ija_b);
  (*iPtr).E += 2.0*trace((*iPtr).V1a_b*T1.t()*T1);
  (*iPtr).E += 2.0*trace((*iPtr).V1aa_b*Wab_ija_b.t()*Wab_ija_b);
  (*iPtr).E += 2.0*trace((*iPtr).V1aa_b*T1.t()*T1);
  (*iPtr).E += 2.0*trace((*iPtr).K1a_b*Wab_ija_b.t()*Wab_ija_b);
  (*iPtr).E += 2.0*trace((*iPtr).K1a_b*T1.t()*T1);
  //t39
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  arma::mat Vab_ija_k=Wab_ija_b*T1.t();
  (*iPtr).E -= 4.0*trace((*iPtr).V2abi_jka*Wab_ija_b*T1.t());
  for(int i=0;i<nels;i++)
  {
    for(int j=0;j<nels;j++)
    {
      n1=i*nact+j;
      n2=j*nact+i;
      n3=j*nels+i;
      n5=i*nels+j;
      for(int a=0;a<virt;a++)
      {
	n4=n3*virt+a;
	n6=n5*virt+a;
	for(int k=0;k<nels;k++)
	{
	  x=Vab_ija_k(n6,k);
	  n7=k*nact+a+nels;
	  n8=(a+nels)*nact+k;
	  D2ab(n1,n7)-=x;
	  D2ab(n7,n1)-=x;
	  D2ab(n2,n8)-=x;
	  D2ab(n8,n2)-=x;
	  //x=V((i*nels+j)*virt+a,k);
	  //D2ab(i*nact+j,k*nact+a+nels)-=x;
	  //D2ab(k*nact+a+nels,i*nact+j)-=x;
	  //D2ab(j*nact+i,(a+nels)*nact+k)-=x;
	  //D2ab((a+nels)*nact+k,j*nact+i)-=x;
	}
      }
    }
  }
  //t40
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  U=(*iPtr).V2abi_jka*Wab_ija_b;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m] -= 4.0*U(m,e);
    }
  }
  //t41
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  U=Wab_ija_b*((*iPtr).K1a_b+(*iPtr).V1a_b+(*iPtr).V1aa_b);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[n1a+n2aa+(e*virt+e)*T2ab.n_rows+m*nels+m] += 4.0*U((m*nels+m)*virt+e,e);
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels && f*nels+n<e*nels+m;n++)
	{
	  x=2.0*( U((m*nels+n)*virt+f,e)+U((n*nels+m)*virt+e,f));
	  grad[n1a+n2aa+(e*virt+f)*T2ab.n_rows+m*nels+n] += x;
	  grad[n1a+n2aa+(f*virt+e)*T2ab.n_rows+n*nels+m] += x;
	}
      }
    }
  }
  //t42
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int m=0;m<nels;m++)
  {
    for(int n=0;n<nels;n++)
    {
      n2=m*nels+n;
      n3=n*nels+m;
      for(int e=0;e<virt;e++)
      {
	n1=n2*virt+e;
	Y = Wab_ija_b.row(n1)*Wab_ija_b.row(n1).t();
	//Y = W.row((m*nels+n)*virt+e)*W.row((m*nels+n)*virt+e).t();
	for(int f=0;f<virt;f++)
	{
	  F2abM(n2,f*virt+e) += 2.0*Y(0);
	  F2abM(n3,e*virt+f) += 2.0*Y(0);
	  //F2abM(m*nels+n,f*virt+e) += 2.0*Y(0);
	  //F2abM(n*nels+m,e*virt+f) += 2.0*Y(0);
	}
      }
    }
  }
  //t43
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int a=0;a<virt;a++)
  {
    Y = Wab_ija_b.col(a).t()*Wab_ija_b.col(a);
    for(int m=0;m<nels;m++)
    {
      F1m(m,a) -= Y(0);
    }
  }
  //t44
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Waa_ija_b(nels*(nels-1)/2*virt,virt); 
//W_jka_b = T2aa_jk_ab
  for(int j=0;j<nels;j++)
  {
    for(int k=j+1;k<nels;k++)
    {
      n1 = indexD(j,k,nels);
      for(int a=0;a<virt;a++)
      {
	n3=n1*virt+a;
	Waa_ija_b(n3,a)=0.0;
	//W(indexD(j,k,nels)*virt+a,a)=0.0;
	for(int b=a+1;b<virt;b++)
	{
	  x=T2aa(n1,indexD(a,b,virt));
	  Waa_ija_b(n3,b)=-x;
	  Waa_ija_b(n1*virt+b,a)=x;
	  //W(indexD(j,k,nels)*virt+a,b)=-T2aa(indexD(j,k,nels),indexD(a,b,virt));
	  //W(indexD(j,k,nels)*virt+b,a)=T2aa(indexD(j,k,nels),indexD(a,b,virt));
	}
      }
    }
  }
  (*iPtr).E += 2.0*trace((*iPtr).V1a_b*Waa_ija_b.t()*Waa_ija_b);
  (*iPtr).E += 2.0*trace((*iPtr).V1aa_b*Waa_ija_b.t()*Waa_ija_b);
  (*iPtr).E += 2.0*trace((*iPtr).K1a_b*Waa_ija_b.t()*Waa_ija_b);
  //t45
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  arma::mat Vaa_ija_k=Waa_ija_b*T1.t();
  (*iPtr).E -= 2.0*trace((*iPtr).V2aai_jka*Waa_ija_b*T1.t());
  for(int i=0;i<nels;i++)
  {
    for(int a=0;a<virt;a++)
    {
      n1=indexD(i,a+nels,nact);
      for(int j=0;j<nels;j++)
      {
	for(int k=j+1;k<nels;k++)
	{
	  x=Vaa_ija_k(indexD(j,k,nels)*virt+a,i);
	  D2aa(indexD(j,k,nact),n1)-=x;
	  D2aa(n1,indexD(j,k,nact))-=x;
	  //D2aa(indexD(j,k,nact),indexD(i,a+nels,nact))-=x;
	  //D2aa(indexD(i,a+nels,nact),indexD(j,k,nact))-=x;
	}
      }
    }
  }
  //t46
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  U=((*iPtr).K1a_b+(*iPtr).V1a_b+(*iPtr).V1aa_b)*Waa_ija_b.t();
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      n1=indexD(m,n,nels);
      for(int e=0;e<virt;e++)
      {
	n2=n1*virt+e;
	for(int f=e+1;f<virt;f++)
	{
	  grad[n1a+indexD(e,f,virt)*T2aa.n_rows+n1] += 4.0*U(e,n1*virt+f);
	  grad[n1a+indexD(e,f,virt)*T2aa.n_rows+n1] -= 4.0*U(f,n2);
	}
      }
    }
  }
  //t47
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  U=(*iPtr).V2aai_jka*Waa_ija_b;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      grad[e*nels+m] -= 2.0*U(m,e);
    }
  }
  //t48
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int e=0;e<virt;e++)
  {
    Y = Waa_ija_b.col(e).t()*Waa_ija_b.col(e);
    for(int m=0;m<nels;m++)
    {
      F1m(m,e) -= Y(0);
    }
  }
  //t49
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  for(int i=0;i<nels;i++)
  {
    for(int j=i+1;j<nels;j++)
    {
      n1 = indexD(i,j,nels);
      for(int a=0;a<virt;a++)
      {
	n2=n1*virt+a;
	Y = Waa_ija_b.row(n2)*Waa_ija_b.row(n2).t();
	//Y = W.row(indexD(i,j,nels)*virt+a)*W.row(indexD(i,j,nels)*virt+a).t();
	for(int b=0;b<a;b++)
	{
	  F2aaM(n1,indexD(b,a,virt)) += 2.0*Y(0);
	  //F2aaM(indexD(i,j,nels),indexD(b,a,virt)) += 2.0*Y(0);
	}
	for(int b=a+1;b<virt;b++)
	{
	  F2aaM(n1,indexD(a,b,virt)) += 2.0*Y(0);
	  //F2aaM(indexD(i,j,nels),indexD(a,b,virt)) += 2.0*Y(0);
	}
      }
    }
  }
  //t50
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  arma::mat Ui_a(nels,virt);
  arma::mat U2(nels,virt);
  for(int i=0;i<nels;i++)
  {
    for(int a=0;a<virt;a++)
    {
      //Ui_a(i,a) = W1_i_a(a*nels+i);
      //Ui_a(i,a) = T1(i,a)*sqrt(F1m(i,a));
      Ui_a(i,a) = T1(i,a)*sqrt(F1m(i,a))+W1_i_a(a*nels+i);
    }
  }
  //t51
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  D1.submat(0,nels,nels-1,nact-1)=Ui_a;
  D1.submat(nels,0,nact-1,nels-1)=Ui_a.t();

  //(*iPtr).E += 4.0*trace((*iPtr).V1ia*arma::vectorise(Ui_a));
  //(*iPtr).E += 4.0*trace((*iPtr).V1ia*arma::vectorise(W1_i_a));
  (*iPtr).E += 4.0*trace((*iPtr).K1ia*arma::vectorise(T1%sqrt(F1m)));
  //(*iPtr).E += 4.0*trace((*iPtr).V1ia*arma::vectorise(T1%sqrt(F1m)));
  //(*iPtr).E += 4.0*trace((*iPtr).V1aia*arma::vectorise(T1%sqrt(F1m)));
  //(*iPtr).E += 4.0*trace((*iPtr).V1ia*arma::vectorise(Ui_a));
  //(*iPtr).E += 4.0*trace((*iPtr).V1aia*arma::vectorise(Ui_a));
  (*iPtr).E += 4.0*trace((*iPtr).K1ia*(Wab_ia_jb+Waa_ia_jb)*vectorise(T1));
  //(*iPtr).E += 4.0*trace((*iPtr).V1ia*(Wab_ia_jb+Waa_ia_jb)*vectorise(T1));
  //(*iPtr).E += 4.0*trace((*iPtr).V1aia*(Wab_ia_jb+Waa_ia_jb)*vectorise(T1));
  for(int m=0;m<nels;m++)
  {
    for(int e=0;e<virt;e++)
    {
      n1=m*nact+e+nels;
      n2=(e+nels)*nact+m;
      n3=indexD(m,e+nels,nact);
      for(int n=0;n<nels;n++)
      {
	D2ab(n1,m*nact+n)+=Ui_a(n,e);
	D2ab(m*nact+n,n1)+=Ui_a(n,e);
	D2ab(n2,n*nact+m)+=Ui_a(n,e);
	D2ab(n*nact+m,n2)+=Ui_a(n,e);
      }
      for(int n=m+1;n<nels;n++)
      {
	D2aa(indexD(m,n,nact),n3)+=Ui_a(n,e);
	D2aa(n3,indexD(m,n,nact))+=Ui_a(n,e);
	D2aa(indexD(m,n,nact),indexD(n,e+nels,nact))-=Ui_a(m,e);
	D2aa(indexD(n,e+nels,nact),indexD(m,n,nact))-=Ui_a(m,e);
	//D2aa(indexD(m,n,nact),indexD(m,e+nels,nact))+=U(n,e);
	//D2aa(indexD(m,e+nels,nact),indexD(m,n,nact))+=U(n,e);
	//D2aa(indexD(m,n,nact),indexD(n,e+nels,nact))-=U(m,e);
	//D2aa(indexD(n,e+nels,nact),indexD(m,n,nact))-=U(m,e);
      }
    }
  }
  //t52
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();

  (*iPtr).E += 2.0*accu((*iPtr).V2abij_ab%T2ab%sqrt(F2abM));
  for(int m=0;m<nels;m++)
  {
    for(int n=0;n<nels;n++)
    {
      n1=m*nels+n;
      n2=m*nact+n;
      for(int e=0;e<virt;e++)
      {
	for(int f=0;f<virt;f++)
	{
	  n3=(e+nels)*nact+f+nels;
	  n4=e*virt+f;
	  x = T2ab(n1,n4)*sqrt(F2abM(n1,n4));
	  D2ab(n3,n2) += x;
	  D2ab(n2,n3) += x;
	  //x = T2ab(m*nels+n,e*virt+f)*pow(F2abM(m*nels+n,e*virt+f),0.5);
	  //D2ab((e+nels)*nact+f+nels,m*nact+n) += x;
	  //D2ab(m*nact+n,(e+nels)*nact+f+nels) += x;
	}
      }
    }
  }
  //t53
  t2 = std::chrono::high_resolution_clock::now();
  time_span = duration_cast<duration<double> >(t2-t1);
  if((*iPtr).ts.size() < nT+1)
  {
    (*iPtr).ts.push_back(time_span);
  }
  else
  {
    (*iPtr).ts[nT] += time_span;
  }
  nT++;
  t1 = std::chrono::high_resolution_clock::now();
  (*iPtr).E += 2.0*accu((*iPtr).V2aaij_ab%T2aa%sqrt(F2aaM));
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      n3=indexD(m,n,nels);
      n2=indexD(m,n,nact);
      for(int e=0;e<virt;e++)
      {
	for(int f=e+1;f<virt;f++)
	{
	  n1=indexD(e,f,virt);
	  n4=indexD(e+nels,f+nels,nact);
	  x=T2aa(n3,n1)*sqrt(F2aaM(n3,n1));
	  D2aa(n4,n2) += x;
	  D2aa(n2,n4) += x;
	  //D2aa(indexD(e+nels,f+nels,nact),indexD(m,n,nact)) += x;
	  //D2aa(indexD(m,n,nact),indexD(e+nels,f+nels,nact)) += x;
	}
      }
    }
  }
  (*iPtr).E += trace((*iPtr).V2abij_kl*T2ab*T2ab.t());
  (*iPtr).E += trace((*iPtr).V2abab_cd*T2ab.t()*T2ab);

  std::cout << "\tE:\t" << (*iPtr).E << std::endl;;
  return;
}

void numEval(const real_1d_array &R,double &E,void *ptr)
{
  bundleNuc *bPtr=static_cast<bundleNuc*>(ptr);

  for(int i=1;i<(*bPtr).atoms.size();i++)
  {
    if(fabs(R[3*i-3]) > 0.01)
    {
      (*bPtr).atoms[i].x=R[3*i-3];
    }
    else
    {
      (*bPtr).atoms[i].x=0.0;
    }
    if(fabs(R[3*i-2]) > 0.01)
    {
      (*bPtr).atoms[i].y=R[3*i-2];
    }
    else
    {
      (*bPtr).atoms[i].y=0.0;
    }
    if(fabs(R[3*i-1]) > 0.01)
    {
      (*bPtr).atoms[i].z=R[3*i-1];
    }
    else
    {
      (*bPtr).atoms[i].z=0.0;
    }
  }

  std::vector<libint2::Atom> allAtoms;
  genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);

  arma::mat T1,T2aa,T2ab;
  double Ep;
  arma::mat K1,V2aa,V2ab;
  runPara(allAtoms,(*bPtr).basis,(*bPtr).nAOs,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).prodTable,Ep,T1,T2aa,T2ab,K1,V2aa,V2ab,false,false);

  E=Ep;
  (*bPtr).E=E;

  return;
}

void gradEval(const real_1d_array &R,double &E,real_1d_array &grad,void *ptr)
{
  bundleNuc *bPtr=static_cast<bundleNuc*>(ptr);

  for(int i=1;i<(*bPtr).atoms.size();i++)
  {
    if(fabs(R[3*i-3]) > 0.01)
    {
      (*bPtr).atoms[i].x=R[3*i-3];
    }
    else
    {
      (*bPtr).atoms[i].x=0.0;
    }
    if(fabs(R[3*i-2]) > 0.01)
    {
      (*bPtr).atoms[i].y=R[3*i-2];
    }
    else
    {
      (*bPtr).atoms[i].y=0.0;
    }
    if(fabs(R[3*i-1]) > 0.01)
    {
      (*bPtr).atoms[i].z=R[3*i-1];
    }
    else
    {
      (*bPtr).atoms[i].z=0.0;
    }
  }

  std::vector<libint2::Atom> allAtoms;
  genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);

  arma::mat T1,T2aa,T2ab;
  arma::mat K1,V2aa,V2ab;
  double Ep;
  runPara(allAtoms,(*bPtr).basis,(*bPtr).nAOs,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).prodTable,Ep,T1,T2aa,T2ab,K1,V2aa,V2ab,true,true);

  arma::mat D1,D2aa,D2ab;
  int nels=T1.n_rows;
  int virt=T1.n_cols;
  intBundle2 integrals={0.0,0.0,K1,V2aa,V2ab,nels,virt,0,true,0,0,0};
  setMats2(integrals);
  makeD2(T1,T2aa,T2ab,D1,D2aa,D2ab,integrals);

  std::cout << "wtf?" << std::endl;

  int nact=nels+virt;
  for(int m=0;m<nels;m++)
  {
    D1(m,m) += 1.0;
    for(int n=0;n<nels;n++)
    {
      D2ab(m*nact+n,m*nact+n) += 1.0;
    }
    for(int n=m+1;n<nels;n++)
    {
      D2aa(indexD(m,n,nact),indexD(m,n,nact)) += 1.0;
    }
  }

  std::cout << 2.0*trace(K1*D1)+trace(V2aa*D2aa)+trace(V2ab*D2ab) << std::endl;  
  std::cout << 2.0*trace(K1*D1) << std::endl;
  std::cout << trace(V2aa*D2aa) << std::endl;
  std::cout << trace(V2ab*D2ab) << std::endl;
  

  return;
}

void setMats2(intBundle2 &i)
{
  int nels=i.nels;
  int virt=i.virt;
  int nact=i.nels+i.virt;
  double x;
  int n1,n2,n3;
  i.Ehf=0.0;

  i.K1i_j=i.K1.submat(0,0,nels-1,nels-1);
  i.K1a_b=i.K1.submat(nels,nels,nact-1,nact-1);
  i.K1ia.resize(1,nels*virt);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      i.K1ia(0,e*nels+m)=i.K1(m,e+nels);
    }
  }
  for(int m=0;m<nels;m++)
  {
    i.Ehf += 2.0*i.K1i_j(m,m);
  }
  //i.V1i_j.resize(nels,nels);
  for(int m=0;m<nels;m++)
  {
    for(int n=0;n<nels;n++)
    {
      x=0.0;
      for(int k=0;k<nels;k++)
      {
	x+=i.V2ab(n*nact+k,m*nact+k); 
      }
      i.K1i_j(n,m)+=x;
    }
  }
  //i.V1ai_j.resize(nels,nels);
  for(int m=0;m<nels;m++)
  {
    for(int n=0;n<nels;n++)
    {
      x=0.0;
      for(int k=0;k<std::min(m,n);k++)
      {
	x += i.V2aa(indexD(k,m,nact),indexD(k,n,nact));
      }
      for(int k=m+1;k<n;k++)
      {
	x -= i.V2aa(indexD(m,k,nact),indexD(k,n,nact));
      }
      for(int k=n+1;k<m;k++)
      {
	x -= i.V2aa(indexD(k,m,nact),indexD(n,k,nact));
      }
      for(int k=std::max(m,n)+1;k<nels;k++)
      {
	x += i.V2aa(indexD(m,k,nact),indexD(n,k,nact));
      }
      //i.V1i_j(m,n)+=0.5*x;
      i.K1i_j(m,n)+=0.5*x;
    }
  }
  //i.V1ia.resize(1,nels*virt);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      x=0.0;
      for(int n=0;n<nels;n++)
      {
	x+=i.V2ab(m*nact+n,(e+nels)*nact+n);
      }
      i.K1ia(0,e*nels+m)+=x;
    }
  }
  /*i.V1i_a.resize(nels,virt);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      x=0.0;
      for(int n=0;n<nels;n++)
      {
	x += i.V2ab(m*nact+n,(e+nels)*nact+n);
      }
      i.V1i_a(m,e) = x;
    }
  }*/
  //i.V1aia.resize(1,nels*virt);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      x=0.0;
      for(int n=0;n<m;n++)
      {
	x += i.V2aa(indexD(n,m,nact),indexD(n,e+nels,nact));
      }
      for(int n=m+1;n<nels;n++)
      {
	x -= i.V2aa(indexD(m,n,nact),indexD(n,e+nels,nact));
      }
      i.K1ia(0,e*nels+m) += 0.5*x;
    }
  }
  i.V2abi_jka.resize(nels,nels*nels*virt);
  for(int j=0;j<nels;j++)
  {
    for(int k=0;k<nels;k++)
    {
      for(int e=0;e<virt;e++)
      {
	for(int l=0;l<nels;l++)
	{
	  i.V2abi_jka(l,(j*nels+k)*virt+e) = i.V2ab(l*nact+e+nels,j*nact+k);
	}
      }
    }
  }
  i.V2aai_jka.resize(nels,nels*(nels-1)/2*virt);
  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      n1=indexD(m,n,nact);
      n3=indexD(m,n,nels);
      for(int e=0;e<virt;e++)
      {
	n2=n3*virt+e;
	for(int k=0;k<nels;k++)
	{
	  i.V2aai_jka(k,n2) = i.V2aa(indexD(k,e+nels,nact),n1);
	}
      }
    }
  }
  i.V1a_b.resize(virt,virt);
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {
      x=0.0;
      for(int m=0;m<nels;m++)
      {
	x += i.V2ab(m*nact+f+nels,m*nact+e+nels);
      }
      i.V1a_b(f,e) = x;
    }
  }
  i.V1aa_b.resize(virt,virt);
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {
      x=0.0;
      for(int m=0;m<nels;m++)
      {
	x += i.V2aa(indexD(m,e+nels,nact),indexD(m,f+nels,nact));
      }
      i.V1aa_b(f,e) = 0.5*x;
    }
  }
  i.V2abia_jb.resize(nels*virt,nels*virt);
  for(int b=0;b<virt;b++)
  {
    for(int j=0;j<nels;j++)
    {
      for(int a=0;a<virt;a++)
      {
	for(int k=0;k<nels;k++)
	{
	  i.V2abia_jb(a*nels+k,b*nels+j) = i.V2ab((a+nels)*nact+j,k*nact+b+nels);
	}
      }
    }
  }
  i.V2abia_jb2.resize(nels*virt,nels*virt);
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      for(int f=0;f<virt;f++)
      {
	for(int n=0;n<nels;n++)
	{
	  i.V2abia_jb2(f*nels+n,e*nels+m) = i.V2ab(m*nact+f+nels,n*nact+e+nels);
	      //x += (*iPtr).V2ab(i*nact+f+nels,m*nact+a+nels)*T2ab(i*nels+n,e*virt+a);
	}
      }
    }
  }
  i.V2aaia_jb.resize(nels*virt,nels*virt);
  for(int b=0;b<virt;b++)
  {
    for(int j=0;j<nels;j++)
    {
      for(int a=0;a<virt;a++)
      {
	for(int k=0;k<nels;k++)
	{
	  i.V2aaia_jb(b*nels+j,a*nels+k) = 0.5*i.V2aa(indexD(j,a+nels,nact),indexD(k,b+nels,nact));	  
	}
      }
    }
  }
  i.V2abh_fgn.resize(virt,virt*virt*nels);
  for(int h=0;h<virt;h++)
  {
    for(int n=0;n<nels;n++)
    {
      for(int f=0;f<virt;f++)
      {
	for(int g=0;g<virt;g++)
	{
	  i.V2abh_fgn(h,(f*virt+g)*nels+n) = i.V2ab((f+nels)*nact+g+nels,n*nact+h+nels);
	}
      }
    }
  }
  i.V2aah_fgn.resize(virt,virt*(virt-1)/2*nels);
  for(int f=0;f<virt;f++)
  {
    for(int g=f+1;g<virt;g++)
    {
      n1=indexD(f,g,virt);
      n3=indexD(f+nels,g+nels,nact);
      for(int m=0;m<nels;m++)
      {
	n2=n1*nels+m;
	for(int h=0;h<virt;h++)
	{
	  i.V2aah_fgn(h,n2) = i.V2aa(indexD(m,h+nels,nact),n3);
	}
      }
    }
  }
  i.V2aaij_kl.resize(nels*(nels-1)/2,nels*(nels-1)/2);
  for(int j=0;j<nels;j++)
  {
    for(int k=j+1;k<nels;k++)
    {
      for(int l=0;l<nels;l++)
      {
	for(int m=l+1;m<nels;m++)
	{
	  i.V2aaij_kl(indexD(l,m,nels),indexD(j,k,nels)) = i.V2aa(indexD(l,m,nact),indexD(j,k,nact));
	}
      }
    }
  }
  i.V2aaab_cd.resize(virt*(virt-1)/2,virt*(virt-1)/2);
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1=indexD(e,f,virt);
      n2=indexD(e+nels,f+nels,nact);
      for(int g=0;g<virt;g++)
      {
	for(int h=g+1;h<virt;h++)
	{
	  i.V2aaab_cd(indexD(g,h,virt),n1) = i.V2aa(indexD(g+nels,h+nels,nact),n2);
	}
      }
    }
  }
  i.V2abij_kl.resize(nels*nels,nels*nels);
  for(int m=0;m<nels;m++)
  {
    for(int j=0;j<nels;j++)
    {
      for(int k=0;k<nels;k++)
      {
	for(int l=0;l<nels;l++)
	{
	  i.V2abij_kl(k*nels+l,m*nels+j) = i.V2ab(k*nact+l,m*nact+j);
	}
      }
    }
  }
  i.V2abab_cd.resize(virt*virt,virt*virt);
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {
      for(int g=0;g<virt;g++)
      {
	for(int h=0;h<virt;h++)
	{
	  i.V2abab_cd(g*virt+h,e*virt+f) = i.V2ab((g+nels)*nact+h+nels,(e+nels)*nact+f+nels);
	}
      }
    }
  }
  i.V2abij_ab.resize(nels*nels,virt*virt);
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {
      for(int m=0;m<nels;m++)
      {
	for(int n=0;n<nels;n++)
	{
	  i.V2abij_ab(m*nels+n,e*virt+f) = i.V2ab(m*nact+n,(e+nels)*nact+f+nels);
	}
      }
    }
  }
  i.V2aaij_ab.resize(nels*(nels-1)/2,virt*(virt-1)/2);
  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  i.V2aaij_ab(indexD(m,n,nels),indexD(e,f,virt)) = i.V2aa(indexD(m,n,nact),indexD(e+nels,f+nels,nact));
	}
      }
    }
  }

  return;
}
