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

void findLabel(std::string label,std::vector<std::string> tLabel,int &nT);
void readIntsGAMESS(std::string name,int nAO,arma::mat &S,arma::mat &K1);
void readInts2GAMESS(std::string name,int nAO,arma::mat &V2ab,arma::mat &V2aa);

void genInts(double &Enuc,const std::vector<libint2::Atom> &atoms,std::string basis,std::vector<arma::mat> &Ss,std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V2s,const std::vector<std::vector<std::vector<orbPair> > > &ao2mo,const std::vector<std::vector<std::vector<int> > > &bas,std::vector<int> nAOs,int &nels,arma::Mat<int> prodTable);
void genIntsDer(double &dEnuc,std::vector<libint2::Atom> &atomsUnique,std::string basis,std::vector<arma::mat> &Ss,std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V,const std::vector<std::vector<std::vector<orbPair> > > &ao2mo,const std::vector<std::vector<std::vector<int> > > &bas,std::vector<int> nAOs,arma::Mat<int> prodTable,std::string pointGroup,std::vector<int> mode,double dx);
void printIts(const real_1d_array &x,double func,void *ptr);
void printItsNuc(const real_1d_array &x,double func,void *ptr);
void evalE(const real_1d_array &T,double &E,real_1d_array &grad,void *ptr);
void runPara(const std::vector<libint2::Atom> &atoms,std::string basis,const std::vector<int> &nAOs,const std::vector<std::vector<std::vector<orbPair> > > &ao2mo,const std::vector<std::vector<std::vector<int> > > &bas,const arma::Mat<int> &prodTable,double &E,arma::mat &T1,arma::mat &T2aa,arma::mat &T2ab,std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V2s,std::vector<arma::mat> &Cs,std::vector<int> &nOcc,std::vector<int> &nCore,bool needT,bool print,int core,int &numPe,bool useOld,char *punchName,bool &fail);

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
  std::vector<int> nCore;
  bool fail;
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
  std::vector<std::vector<int> > modes;
  int core;
  int numIt;
  double E;
  std::vector<std::chrono::duration<double> > ts;
  std::vector<std::string> tLabel;
  double oldE;
  std::vector<double> g;
  arma::mat T1,T2aa,T2ab;
  std::vector<arma::mat> Cs;
  std::vector<int> nOcc,nCore;
  int numE,numPe;
  bool fail;
  char *punchName;
};

void numEval(const real_1d_array &R,double &E,real_1d_array &grad,void *ptr);
void gradEval(const real_1d_array &R,double &E,real_1d_array &grad,void *ptr);
void gradEvalU(const real_1d_array &R,double &E,real_1d_array &grad,void *ptr);
void gradEvalZ(const real_1d_array &R,double &E,real_1d_array &grad,void *ptr);

void makeD2(const arma::mat &T1,const arma::mat &T2aa,const arma::mat &T2ab,arma::mat &D1,arma::mat &D2aa,arma::mat &D2ab);
void solveU(std::vector<double> eps,const arma::mat &dK1,const arma::mat &dV2ab,const arma::mat &dS,const arma::mat &V2ab,arma::mat &U,int nels);
void solveUsym(std::vector<std::vector<double> > eps,const std::vector<arma::mat> &dK1,const std::vector<std::vector<arma::mat> > &dV2ab,const std::vector<arma::mat> &dS,const std::vector<std::vector<arma::mat> > &V2ab,std::vector<arma::mat> &U,std::vector<int> nOcc,std::vector<arma::mat> &B);
void makeKt(const arma::mat &U,arma::mat &K1t,arma::mat &V2aaT,arma::mat &V2abT,const arma::mat &K1,const arma::mat &V2aa,const arma::mat &V2ab);
void makeKtsym(const std::vector<arma::mat> &U,std::vector<arma::mat> &K1t,std::vector<std::vector<arma::mat> > &V2t,const std::vector<arma::mat> &K1s,const std::vector<std::vector<arma::mat> > &V2s);

void rotateBack(const std::vector<arma::mat> &K1s,const std::vector<std::vector<arma::mat> > &V2s,std::vector<int> nOcc,arma::mat &K1b,arma::mat &V2aa,arma::mat &V2ab);
void rotateForward(const std::vector<int> &nOcc,const arma::mat &D1,arma::mat &D2aa,const arma::mat &D2ab,const std::vector<int> &nMO,std::vector<arma::mat> &D1s,std::vector<std::vector<arma::mat> > &D2aaS,std::vector<std::vector<arma::mat> > &D2abS,const std::vector<std::vector<std::vector<int> > > &bas,const std::vector<int> &nCore);
void rotateMO_AO(std::vector<arma::mat> &D1,std::vector<std::vector<arma::mat> > &D2aa,std::vector<std::vector<arma::mat> > &D2ab,const std::vector<arma::mat> &Cs,const std::vector<int> &nMO);
void makeV2aa(const std::vector<std::vector<arma::mat> > &V2ab,std::vector<std::vector<arma::mat> > &V2aaS,const std::vector<std::vector<std::vector<int> > > &bas,std::vector<int> nMOs);
void makeZ(const std::vector<arma::mat> &D1,const std::vector<std::vector<arma::mat> > &D2aa,const std::vector<std::vector<arma::mat> > &D2ab,const std::vector<arma::mat> &K1,const std::vector<std::vector<arma::mat> > &V2aa,const std::vector<std::vector<arma::mat> > &V2ab,const std::vector<std::vector<double> > &eps,const std::vector<int> &nMO,const std::vector<int> &nOcc,arma::vec &X,arma::vec &Z);
void makeZ2(const std::vector<arma::mat> &D1,const std::vector<std::vector<arma::mat> > &D2aa,const std::vector<std::vector<arma::mat> > &D2ab,const std::vector<arma::mat> &K1,const std::vector<std::vector<arma::mat> > &V2aa,const std::vector<std::vector<arma::mat> > &V2ab,const std::vector<std::vector<double> > &eps,const std::vector<int> &nMO,const std::vector<int> &nOcc,arma::vec &dX,arma::vec &Z,std::vector<std::vector<std::vector<int> > > &nonDeg);
void makeB(const std::vector<std::vector<double> > &eps,const std::vector<arma::mat> &Cs,const std::vector<arma::mat> &S,const std::vector<arma::mat> &K1,const std::vector<std::vector<arma::mat> > &V2,const std::vector<int> &nMO,const std::vector<int> &nOcc,const std::vector<std::vector<arma::mat> > &V2s,arma::vec &B,arma::vec &dS,const std::vector<std::vector<std::vector<int> > > &nonDeg);

void readInputFile(char *fileName,char *xyzName,char *punchName,std::string &basis,std::string &pointGroup,int &method);

int main(int argc, char *argv[])
{
  int nMO=195;
  int nAO=195;

  char cName[]="/home/andrew/p2rdm/punch/170228/is8.dat";
  char logName[]="/home/andrew/p2rdm/logs/170228/is8.out";

  std::ifstream cFile,logFile;
  cFile.open(cName);

  arma::mat C1(nAO,nMO);
  arma::mat C2(nAO,nMO);
  arma::mat S;

  cFile.close();

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

  double max=0;
  double rms=0;
  for(int m=0;m<bPtr->g.size();m++)
  {
    rms += (bPtr->g[m]*bPtr->g[m]);
    if(fabs(bPtr->g[m]) > max)
    {
      max = fabs(bPtr->g[m]);
    }
  }
  rms /= bPtr->g.size();
  rms = sqrt(rms);

  printf("\nIter\tE\t\tdE\t\t\tmax grad\t\trms grad\t\tnumE\tnumPe\n");
  printf("%d\t%.9f\t%.10E\t%.10E\t%.10E\t%d\t%d\n",bPtr->numIt+1,bPtr->E,bPtr->E-bPtr->oldE,max,rms,bPtr->numE,bPtr->numPe/bPtr->numE);
  std::cout << "\nUnique atoms (angstrom):" << std::endl;
  std::cout << "Z\tx\t\ty\t\tz\n";

  double bohr= 0.52917721092;

  for(int n=0;n<(*bPtr).atoms.size();n++)
  {
    printf("%d\t%.9f\t%.9f\t%.9f\n",bPtr->atoms[n].atomic_number,bPtr->atoms[n].x*bohr,bPtr->atoms[n].y*bohr,bPtr->atoms[n].z*bohr);
  }

  (*bPtr).numIt++;
  bPtr->oldE = bPtr->E;
  bPtr->numE=bPtr->numPe=0;
  std::cout << std::endl;

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

  iPtr->fail=false;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      //F1m(m,e) = fabs(F1m(m,e));
      if(F1m(m,e) < 0.0)
      {
	//F1m(m,e)=fabs(F1m(m,e));
	iPtr->fail=true;
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
	  //F2aaM(indexD(m,n,nels),indexD(e,f,virt)) = fabs(F2aaM(indexD(m,n,nels),indexD(e,f,virt)));
	  if(F2aaM(indexD(m,n,nels),indexD(e,f,virt)) < 0.0)
	  {
	    //F2aaM(indexD(m,n,nels),indexD(e,f,virt))=fabs(F2aaM(indexD(m,n,nels),indexD(e,f,virt)));
	    iPtr->fail=true;
	  }
	}
      }
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int f=0;f<virt;f++)
    {
      for(int m=0;m<nels;m++)
      {
	for(int n=0;n<nels;n++)
	{
	  if(F2abM(m*nels+n,e*virt+f) < 0.0)
	  {
	    //F2abM(m*nels+n,e*virt+f)=fabs(F2abM(m*nels+n,e*virt+f));
	    iPtr->fail=true;
	  }
	}
      }
    }
  }

  if(iPtr->fail)
  {
    return;
  } 

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
  std::vector<int> nc=i.nCore;
  for(int p=0;p<i.K1s.size();p++)
  {
    nact += i.K1s[p].n_cols-nc[p];
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
    nVirt.push_back(i.K1s[p].n_cols-nOcc[p]-nc[p]);
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
	i.K1i_j(o1+m,o1+n)=i.K1s[p](m+nc[p],n+nc[p]);
	x=0.0;
	for(int q=0;q<i.K1s.size();q++)
	{
	  for(int n3=0;n3<nc[q];n3++)
	  {
	    x += 2.0*i.V2s[q*i.K1s.size()+p][q*i.K1s.size()+p](n3*nTot[p]+m+nc[p],n3*nTot[p]+n+nc[p]);
	    x -= i.V2s[p*i.K1s.size()+q][q*i.K1s.size()+p]((m+nc[p])*nTot[q]+n3,n3*nTot[p]+n+nc[p]);
	  }
	}
	i.K1i_j(o1+m,o1+n) += x;
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
	i.K1a_b(v1+b,v1+a)=i.K1s[p](b+nOcc[p]+nc[p],a+nOcc[p]+nc[p]);
	x=0.0;
	for(int q=0;q<i.K1s.size();q++)
	{
	  for(int n3=0;n3<nc[q];n3++)
	  {
	    x += 2.0*i.V2s[q*i.K1s.size()+p][q*i.K1s.size()+p](n3*nTot[p]+b+nOcc[p]+nc[p],n3*nTot[p]+a+nOcc[p]+nc[p]);
	    x -= i.V2s[p*i.K1s.size()+q][q*i.K1s.size()+p]((b+nOcc[p]+nc[p])*nTot[q]+n3,n3*nTot[p]+a+nOcc[p]+nc[p]);
	  }
	}
	i.K1a_b(v1+b,v1+a) += x;
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
	i.K1ia(0,(a+v1)*nels+m+o1)=i.K1s[p](m+nc[p],a+nOcc[p]+nc[p]);
	x=0.0;
	for(int q=0;q<i.K1s.size();q++)
	{
	  for(int n3=0;n3<nc[q];n3++)
	  {
	    x += 2.0*i.V2s[q*i.K1s.size()+p][q*i.K1s.size()+p](n3*nTot[p]+m+nc[p],n3*nTot[p]+a+nOcc[p]+nc[p]);
	    x -= i.V2s[p*i.K1s.size()+q][q*i.K1s.size()+p]((m+nc[p])*nTot[q]+n3,n3*nTot[p]+a+nOcc[p]+nc[p]);
	  }
	}
	i.K1ia(0,(a+v1)*nels+m+o1) += x;
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
	    x += 2.0*i.V2s[p*i.K1s.size()+q][p*i.K1s.size()+q]((n+nc[p])*nTot[q]+t+nc[q],(m+nc[p])*nTot[q]+t+nc[q]);
	    x -= i.V2s[q*i.K1s.size()+p][p*i.K1s.size()+q]((t+nc[q])*nTot[p]+n+nc[p],(m+nc[p])*nTot[q]+t+nc[q]);
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
	    x += 2.0*i.V2s[p*nOcc.size()+q][p*nOcc.size()+q]((m+nc[p])*nTot[q]+n+nc[q],(e+nOcc[p]+nc[p])*nTot[q]+n+nc[q]);
	    x -= i.V2s[q*nOcc.size()+p][p*nOcc.size()+q]((n+nc[q])*nTot[p]+m+nc[p],(e+nOcc[p]+nc[p])*nTot[q]+n+nc[q]);
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
		    i.V2abi_jka(l+o4,((j+o1)*nels+k+o2)*virt+e+v3) = i.V2s[s*nOcc.size()+r][p*nOcc.size()+q]((l+nc[s])*nTot[r]+e+nOcc[r]+nc[r],(j+nc[p])*nTot[q]+k+nc[q]);
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
	    x += 2.0*i.V2s[p*nOcc.size()+q][p*nOcc.size()+q]((f+nOcc[p]+nc[p])*nTot[q]+m+nc[q],(e+nOcc[p]+nc[p])*nTot[q]+m+nc[q]);
	    x -= i.V2s[q*nOcc.size()+p][p*nOcc.size()+q]((m+nc[q])*nTot[p]+f+nOcc[p]+nc[p],(e+nOcc[p]+nc[p])*nTot[q]+m+nc[q]);
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
		    i.V2abia_jb((a+v3)*nels+k+o4,(b+v1)*nels+j+o2) = i.V2s[r*nOcc.size()+q][s*nOcc.size()+p]((a+nOcc[r]+nc[r])*nTot[q]+j+nc[q],(k+nc[s])*nTot[p]+b+nOcc[p]+nc[p]);
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
		    i.V2abia_jb2((a+v3)*nels+k+o4,(b+v1)*nels+j+o2) = i.V2s[q*nOcc.size()+r][s*nOcc.size()+p]((j+nc[q])*nTot[r]+a+nOcc[r]+nc[r],(k+nc[s])*nTot[p]+b+nOcc[p]+nc[p]);
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
		    i.V2abh_fgn(h+v4,((f+v1)*virt+g+v2)*nels+n+o3) = i.V2s[r*nOcc.size()+s][p*nOcc.size()+q]((n+nc[r])*nTot[s]+h+nOcc[s]+nc[s],(f+nOcc[p]+nc[p])*nTot[q]+g+nOcc[q]+nc[q]);
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
		    i.V2abij_kl((k+o3)*nels+l+o4,(m+o1)*nels+j+o2) = i.V2s[r*nOcc.size()+s][p*nOcc.size()+q]((k+nc[r])*nTot[s]+l+nc[s],(m+nc[p])*nTot[q]+j+nc[q]);
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
		    i.V2abab_cd((g+v3)*virt+h+v4,(e+v1)*virt+f+v2) = i.V2s[r*nOcc.size()+s][p*nOcc.size()+q]((g+nOcc[r]+nc[r])*nTot[s]+h+nOcc[s]+nc[s],(e+nOcc[p]+nc[p])*nTot[q]+f+nOcc[q]+nc[q]);
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
		    i.V2abij_ab((m+o3)*nels+n+o4,(e+v1)*virt+f+v2) = i.V2s[r*nOcc.size()+s][p*nOcc.size()+q]((m+nc[r])*nTot[s]+n+nc[s],(e+nOcc[p]+nc[p])*nTot[q]+f+nOcc[q]+nc[q]);
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

  //i.K1s.clear();
  //i.V2s.clear();

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

void runPara(const std::vector<libint2::Atom> &atoms,std::string basis,const std::vector<int> &nAOs,const std::vector<std::vector<std::vector<orbPair> > > &ao2mo,const std::vector<std::vector<std::vector<int> > > &bas,const arma::Mat<int> &prodTable,double &E,arma::mat &T1,arma::mat &T2aa,arma::mat &T2ab,std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V2s,std::vector<arma::mat> &Cs,std::vector<int> &nOcc,std::vector<int> &nCore,bool needT,bool print,int core,int &numPe,bool useOld,char *punchName,bool &fail)
{
  double Enuc,Ecore;
  int nels;

  std::chrono::high_resolution_clock::time_point t1,t2,t3;
  std::chrono::duration<double> ts;

  int nAO=0;
  std::vector<arma::mat> Ss;
  //std::vector<std::vector<arma::mat> > V2s;

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
  nOcc.resize(nAOs.size());
  Cs.resize(nAOs.size());
  double Ehf=solveHFsym(nels,Cs,Ss,K1s,V2s,bas,nOcc,nCore,nAOs,print,core,useOld);
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
  rotateInts(K1s,V2s,Cs,bas,nOcc,nCore,Ecore);
  setTime(t1,t2,ts);
  if(print)
  {
    std::cout << "rotation time: " << ts.count() << std::endl << std::endl;;
  }

  std::vector<std::vector<double> > eps(nOcc.size());
  double x;
  for(int p=0;p<nOcc.size();p++)
  {
    for(int i=0;i<K1s[p].n_cols;i++)
    {
      x=K1s[p](i,i);
      for(int q=0;q<nOcc.size();q++)
      {
	for(int j=0;j<nOcc[q]+nCore[q];j++)
	{
	  x += 2.0*V2s[p*nOcc.size()+q][p*nOcc.size()+q](i*K1s[q].n_cols+j,i*K1s[q].n_cols+j);
	  x -= V2s[q*nOcc.size()+p][p*nOcc.size()+q](j*K1s[p].n_cols+i,i*K1s[q].n_cols+j);
	}
      }
      eps[p].push_back(x);
    }
  }
  int nMO=0;
  nAO=0;
  std::vector<int> ns(nOcc.size());
  for(int i=0;i<nOcc.size();i++)
  {
    nMO += eps[i].size();
    nAO += Cs[i].n_rows;
    ns[i]=0;
  }  
  std::vector<std::vector<int> > order(nMO,std::vector<int>(2));
  int n; double min;
  for(int i=0;i<nMO;i++)
  {
    min=1000000000;
    n=0;
    for(int j=0;j<nOcc.size();j++)
    {
      if(ns[j] < K1s[j].n_cols && eps[j][ns[j]] < min)
      {
	min = eps[j][ns[j]];
	n=j;
      }
    }
    order[i][0] = n;
    order[i][1] = ns[n];
    ns[n] += 1;
  }

  std::fstream punchFile;
  punchFile.open(punchName,std::fstream::app);

  int numPer=5;
  int ao;
  std::string str;
  char cstr[100];
  for(int i=0;i<nMO;i+=numPer)
  {
    punchFile << "\n\nMO:";
    for(int j=i;j<std::min(i+numPer,nMO);j++)
    {
      sprintf(cstr,"\t\t\t%d",j+1);
      str=std::string(cstr);
      punchFile << str;
    }
    punchFile << "\nE:";
    for(int j=i;j<std::min(i+numPer,nMO);j++)
    {
      sprintf(cstr,"\t\t\t%.3f",eps[order[j][0]][order[j][1]]);
      str=std::string(cstr);
      punchFile << str;
    }
    punchFile << "\nSym:";
    for(int j=i;j<std::min(i+numPer,nMO);j++)
    {
      sprintf(cstr,"\t\t\t%d",order[j][0]);
      str=std::string(cstr);
      punchFile << str;
    }
    punchFile << "\nAO:";
    ao=0;
    for(int p=0;p<nOcc.size();p++)
    {
      for(int k=0;k<Cs[p].n_rows;k++)
      {
	punchFile << "\n" << ao+1 << "\t\t";
	for(int j=i;j<std::min(i+numPer,nMO);j++)
	{
	  if(order[j][0] == p)
	  {
	    sprintf(cstr,"\t%.10E",Cs[order[j][0]](k,order[j][1]));
	    str=std::string(cstr);
	    punchFile << str;
	  }
	  else
	  {
	    sprintf(cstr,"\t0.0\t\t");
	    str=std::string(cstr);
	    punchFile << str;
	  }
	}
	ao++;
      }
    }
  }
  punchFile << std::endl << std::endl;
  punchFile.close();

  nels=0;
  for(int r=0;r<nOcc.size();r++)
  {
    nels += nOcc[r];
  }

  intBundle integrals={0.0,Enuc,K1s,V2s,nOcc,print,nels};
  integrals.nCore=nCore;
  setMats(integrals);

  nels=integrals.nels;
  int virt=integrals.virt;
  int numTot=nels*virt;
  numTot += nels*(nels-1)/2*virt*(virt-1)/2;
  numTot += nels*nels*virt*virt;
  double* Ts;
  Ts = new double[numTot];
  if(useOld)
  {
    int n1a=nels*virt;
    int n2aa=T2aa.n_rows*T2aa.n_cols;
    for(int e=0;e<virt;e++)
    {
      for(int m=0;m<nels;m++)
      {
	Ts[e*nels+m]=T1(m,e);
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
	    Ts[n1a+indexD(e,f,virt)*T2aa.n_rows+indexD(m,n,nels)]=T2aa(indexD(m,n,nels),indexD(e,f,virt));
	  }
	}
      }
    }
    for(int e=0;e<virt;e++)
    {
      for(int f=0;f<virt;f++)
      {
	for(int m=0;m<nels;m++)
	{
	  for(int n=0;n<nels;n++)
	  {
	    Ts[n1a+n2aa+(e*virt+f)*T2ab.n_rows+m*nels+n]=T2ab(m*nels+n,e*virt+f);
	  }
	}
      }
    }
  }
  else
  {
    for(int i=0;i<numTot;i++)
    {
      Ts[i]=0.0;
    }
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
//  if(print)
  {
    minlbfgsoptimize(state,evalE,printIts,&integrals);
  }
//  else
  {
//    minlbfgsoptimize(state,evalE,NULL,&integrals);
  }
  minlbfgsresults(state,T,rep);
  numPe += integrals.numIt;
  setTime(t1,t2,ts);

  fail=integrals.fail;
  E=integrals.Ehf+integrals.E+integrals.Enuc+Ecore;
  //std::cout << integrals.Ehf+integrals.E+Ecore << std::endl;

  if(print)
  {
    std::cout << "Total Energy:\t";
    std::cout << E << std::endl;// (integrals).Ehf+Ecore+(integrals.E)+integrals.Enuc << std::endl;
    //std::cout << (integrals).Ehf+Ecore+(integrals.E)+integrals.Enuc << std::endl;
    //std::cout << (integrals).Ehf+(integrals.E)+integrals.Enuc << std::endl;
    std::cout << "P2RDM Time: " << ts.count() << std::endl;
  }

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

void makeD2(const arma::mat &T1,const arma::mat &T2aa,const arma::mat &T2ab,arma::mat &D1,arma::mat &D2aa,arma::mat &D2ab)
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

  double y;
  int n1a=nels*virt;
  int n2aa=T2aa.n_rows*T2aa.n_cols;
  arma::mat U,Y;

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

  arma::mat Wab_ij_kl=T2ab*T2ab.t();
  arma::mat W1_i_j=-T1*T1.t();
  for(int m=0;m<nels;m++)
  {
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
	}
      }
      for(int b=a+1;b<virt;b++)
      {
	n2 = indexD(a,b,virt);
	for(int n=0;n<m;n++)
	{
	  n1 = indexD(n,m,nels);
	  F2aaM(n1,n2) += x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  n1= indexD(m,n,nels);
	  F2aaM(n1,n2) += x;
	}
      }
    }
  }
  for(int m=0;m<nels;m++)
  {
    for(int n=0;n<nels;n++)
    {
      n1 = m*nels+n;
      x = Wab_ij_kl(n1,n1);
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
	for(int l=0;l<nels;l++)
	{
	  n4 = k*nact+l;
	  n6 = k*nels+l;
	  D2ab(n1,n4)+=Wab_ij_kl(n5,n6);
	}
      }
    }
  }

  Wab_ij_kl.reset();

  arma::mat Waa_ij_kl=T2aa*T2aa.t();
  for(int i=0;i<nels;i++)
  {
    for(int j=0;j<nels;j++)
    {
      for(int k=0;k<std::min(i,j);k++)
      {
	n1 = indexD(k,i,nels);
	n2 = indexD(k,j,nels);
	W1_i_j(i,j) -= Waa_ij_kl(n1,n2);
      }
      for(int k=i+1;k<j;k++)
      {
	n1 = indexD(i,k,nels);
	n2 = indexD(k,j,nels);
	W1_i_j(i,j) += Waa_ij_kl(n1,n2);
      }
      for(int k=j+1;k<i;k++)
      {
	n1 = indexD(k,i,nels);
	n2 = indexD(j,k,nels);
	W1_i_j(i,j) += Waa_ij_kl(n1,n2);
      }
      for(int k=std::max(i,j)+1;k<nels;k++)
      {
	n1 = indexD(i,k,nels);
	n2 = indexD(j,k,nels);
	W1_i_j(i,j) -= Waa_ij_kl(n1,n2);
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
	}
      }
    }
  }
  D1.submat(0,0,nels-1,nels-1)=W1_i_j;
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

  W1_i_j.reset();

  for(int m=0;m<nels;m++)
  {
    for(int n=m+1;n<nels;n++)
    {
      n1 = indexD(m,n,nels);
      x = Waa_ij_kl(n1,n1);
      for(int e=0;e<virt;e++)
      {
	for(int f=e+1;f<virt;f++)
	{
	  n2=indexD(e,f,virt);
	  F2aaM(n1,n2) -= x;
	}
      }
    }
  }

  Waa_ij_kl.reset();

  arma::mat Wab_ab_cd=T2ab.t()*T2ab;
  arma::mat W1_a_b=T1.t()*T1;
  for(int a=0;a<virt;a++)
  {
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
	}
      }
      for(int n=m+1;n<nels;n++)
      {
	n1 = indexD(m,n,nels);
	for(int b=0;b<a;b++)
	{
	  n2=indexD(b,a,virt);
	  F2aaM(n1,n2) -= x;
	}
	for(int b=a+1;b<virt;b++)
	{
	  n2=indexD(a,b,virt);
	  F2aaM(n1,n2) -= x;
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
      x=0.0;
      for(int g=0;g<virt;g++)
      {
	n3=e*virt+g;
	n4=f*virt+g;
	x += Wab_ab_cd(n3,n4);
	for(int h=0;h<virt;h++)
	{
	  n5=(g+nels)*nact+h+nels;
	  n6=g*virt+h;
	  D2ab(n1,n5) += Wab_ab_cd(n2,n6);
	}
      }
      W1_a_b(e,f) += x;
      x = Wab_ab_cd(n2,n2);
      for(int m=0;m<nels;m++)
      {
	for(int n=0;n<nels;n++)
	{
	  n7=m*nels+n;
	  F2abM(n7,n2) -= x;
	}
      }
    }
  }

  Wab_ab_cd.reset();

  arma::mat Waa_ab_cd=T2aa.t()*T2aa;
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
	}
      }
    }
  }
  D1.submat(nels,nels,nact-1,nact-1)=W1_a_b;
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
	D2aa(n1,indexD(m,f+nels,nact)) += W1_a_b(e,f);
      }
    }
  }

  W1_a_b.reset();

  for(int e=0;e<virt;e++)
  {
    for(int f=e+1;f<virt;f++)
    {
      n1 = indexD(e,f,virt);
      x = Waa_ab_cd(n1,n1);
      for(int m=0;m<nels;m++)
      {
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n1) -= x;
	}
      }
    }
  }

  Waa_ab_cd.reset();

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
	}
      }
    }
  }
  arma::mat Vab_ia_jb2=Wab_ia_jb2*Wab_ia_jb2.t();

  Wab_ia_jb2.reset();

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
	}
      }
    }
  }

  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n1=e*nels+m;
      x = Vab_ia_jb2(n1,n1);
      for(int f=0;f<virt;f++)
      {
	n2=e*virt+f;
	n3=f*virt+e;
	for(int n=0;n<nels;n++)
	{
	  F2abM(m*nels+n,n3) -= x;
	  F2abM(n*nels+m,n2) -= x;
	}
      }
    }
  }

  Vab_ia_jb2.reset();

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
	}
      }
    }
  }
  arma::mat Vab_ia_jb=Wab_ia_jb*Wab_ia_jb.t();
  for(int f=0;f<virt;f++)
  {
    for(int m=0;m<nels;m++)
    {
      n3=f*nels+m;
      for(int e=0;e<virt;e++)
      {
	n1 = indexD(m,e+nels,nact);
	for(int n=0;n<nels;n++)
	{
	  n2 = indexD(n,f+nels,nact);
	  D2aa(n2,n1) -= T1(n,e)*T1(m,f);
	  D2aa(n2,n1) -= Vab_ia_jb(e*nels+n,n3);
	}
      }
    }
  }

  Vab_ia_jb.reset();

  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n1=e*nels+m;
      x=dot(Wab_ia_jb.col(n1),Wab_ia_jb.col(n1));
      F1m(m,e) += x;
      for(int f=0;f<virt;f++)
      {
	n2=e*virt+f;
	n3=f*virt+e;
	for(int n=0;n<nels;n++)
	{
	  F2abM(n*nels+m,n3) -= x;
	  F2abM(m*nels+n,n2) -= x;
	}
      }
      for(int f=0;f<e;f++)
      {
	n2=indexD(f,e,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n2) -= x;
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
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n2) -= x;
	}
      }
    }
  }

  arma::mat Waa_ia_jb(nels*virt,nels*virt);
  for(int a=0;a<virt;a++)
  {
    for(int i=0;i<nels;i++)
    {
      n3=a*nels+i;
      for(int m=0;m<nels;m++)
      {
	Waa_ia_jb(n3,a*nels+m)=0.0;
      }
      for(int f=0;f<a;f++)
      {
	n1=indexD(f,a,virt);
	Waa_ia_jb(n3,f*nels+i)=0.0;
	for(int m=0;m<i;m++)
	{
	  Waa_ia_jb(n3,f*nels+m) = T2aa(indexD(m,i,nels),n1);
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
  arma::mat W1_i_a = (Wab_ia_jb+Waa_ia_jb)*arma::vectorise(T1);

  arma::mat Vaa_ia_jb=Waa_ia_jb.t()*Waa_ia_jb;
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n3=e*nels+m;
      n1=indexD(m,e+nels,nact);
      for(int n=0;n<nels;n++)
      {
	n4=e*nels+n;
	for(int f=0;f<virt;f++)
	{
	  D2aa(indexD(n,f+nels,nact),n1) -= Vaa_ia_jb(f*nels+m,n4);
	}
      }
    }
  }
  for(int e=0;e<virt;e++)
  {
    for(int m=0;m<nels;m++)
    {
      n3=e*nels+m;
      x=Vaa_ia_jb(n3,n3);
      F1m(m,e) += x;
      for(int f=0;f<virt;f++)
      {
	n4=e*virt+f;
	n5=f*virt+e;
	for(int n=0;n<nels;n++)
	{
	  F2abM(m*nels+n,n4) -= x;
	  F2abM(n*nels+m,n5) -= x;
	}
      }
      for(int f=0;f<e;f++)
      {
	n1 = indexD(f,e,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n1) -= x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n1) -= x;
	}
      }
      for(int f=e+1;f<virt;f++)
      {
	n1 = indexD(e,f,virt);
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n1) -= x;
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n1) -= x;
	}
      }
    }
  }

  Vaa_ia_jb.reset();

  arma::mat Uab_ia_jb = Wab_ia_jb*Waa_ia_jb;

  Wab_ia_jb.reset();
  Waa_ia_jb.reset();

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
	  x = T1(n,e)*T1(m,f) + Uab_ia_jb(e*nels+n,n3) + Uab_ia_jb(n3,e*nels+n);
	  D2ab(n1,(f+nels)*nact+n) += x;
	  D2ab(n2,n*nact+f+nels) += x;
	}
      }
    }
  }

  Uab_ia_jb.reset();

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
	}
      }
    }
  }

  arma::mat Vab_a_jbc = T1.t()*Wab_i_jab;
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
	}
      }
    }
  }

  Vab_a_jbc.reset();

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
	for(int n=0;n<nels;n++)
	{
	  F2abM(m*nels+n,n3) += 2.0*Y(0);
	  F2abM(n*nels+m,n4) += 2.0*Y(0);
	}
      }
    }
  }
  for(int m=0;m<nels;m++)
  {
    Y = Wab_i_jab.row(m)*Wab_i_jab.row(m).t();
    for(int e=0;e<virt;e++)
    {
      F1m(m,e) -= Y(0);
    }
  }

  Wab_i_jab.reset();

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
	for(int i=0;i<m;i++)
	{
	  Waa_i_jab(i,n3) = -T2aa(indexD(i,m,nels),n2);
	}
	for(int i=m+1;i<nels;i++)
	{
	  Waa_i_jab(i,n3) = T2aa(indexD(m,i,nels),n2);
	}
      }
    }
  }
  arma::mat Vaa_a_jbc = T1.t()*Waa_i_jab;
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
	  n2=indexD(m,e+nels,nact);
	  D2aa(n1,n2) += x;
	  D2aa(n2,n1) += x;
	}
      }
    }
  }

  Vaa_a_jbc.reset();

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
	for(int n=0;n<m;n++)
	{
	  F2aaM(indexD(n,m,nels),n2) += 2.0*Y(0);
	}
	for(int n=m+1;n<nels;n++)
	{
	  F2aaM(indexD(m,n,nels),n2) += 2.0*Y(0);
	}
      }
    }
  }

  Waa_i_jab.reset();

  arma::mat Wab_ija_b(nels*nels*virt,virt);
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
	}
      }
    }
  }
  arma::mat Vab_ija_k=Wab_ija_b*T1.t();
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
	}
      }
    }
  }

  Vab_ija_k.reset();

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
	for(int f=0;f<virt;f++)
	{
	  F2abM(n2,f*virt+e) += 2.0*Y(0);
	  F2abM(n3,e*virt+f) += 2.0*Y(0);
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

  arma::mat Waa_ija_b(nels*(nels-1)/2*virt,virt); 
  for(int j=0;j<nels;j++)
  {
    for(int k=j+1;k<nels;k++)
    {
      n1 = indexD(j,k,nels);
      for(int a=0;a<virt;a++)
      {
	n3=n1*virt+a;
	Waa_ija_b(n3,a)=0.0;
	for(int b=a+1;b<virt;b++)
	{
	  x=T2aa(n1,indexD(a,b,virt));
	  Waa_ija_b(n3,b)=-x;
	  Waa_ija_b(n1*virt+b,a)=x;
	}
      }
    }
  }
  arma::mat Vaa_ija_k=Waa_ija_b*T1.t();
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
	}
      }
    }
  }

  Vaa_ija_k.reset();

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
	for(int b=0;b<a;b++)
	{
	  F2aaM(n1,indexD(b,a,virt)) += 2.0*Y(0);
	}
	for(int b=a+1;b<virt;b++)
	{
	  F2aaM(n1,indexD(a,b,virt)) += 2.0*Y(0);
	}
      }
    }
  }

  Waa_ija_b.reset();

  arma::mat Ui_a(nels,virt);
  for(int i=0;i<nels;i++)
  {
    for(int a=0;a<virt;a++)
    {
      Ui_a(i,a) = T1(i,a)*sqrt(F1m(i,a))+W1_i_a(a*nels+i);
    }
  }

  W1_i_a.reset();
  D1.submat(0,nels,nels-1,nact-1)=Ui_a;
  D1.submat(nels,0,nact-1,nels-1)=Ui_a.t();

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
      }
    }
  }

  Ui_a.reset();
  
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
	}
      }
    }
  }
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
	}
      }
    }
  }

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

  return;
}

void numEval(const real_1d_array &R,double &E,real_1d_array &grad,void *ptr)
{
  int num;
  bundleNuc *bPtr=static_cast<bundleNuc*>(ptr);
  for(int m=0;m<bPtr->modes.size();m++)
  {
    switch((*bPtr).modes[m][1])
    {
      case 0:
	(*bPtr).atoms[(*bPtr).modes[m][0]].x = R[m];
	break;
      case 1:
	(*bPtr).atoms[(*bPtr).modes[m][0]].y = R[m];
	break;
      case 2:
	(*bPtr).atoms[(*bPtr).modes[m][0]].z = R[m];
	break;
    }
  }
  bPtr->numE++;

  std::cout << "\nUnique atoms (angstrom):" << std::endl;
  std::cout << "Z\tx\t\ty\t\tz\n";

  double bohr= 0.52917721092;

  for(int n=0;n<(*bPtr).atoms.size();n++)
  {
    printf("%d\t%.9f\t%.9f\t%.9f\n",bPtr->atoms[n].atomic_number,bPtr->atoms[n].x*bohr,bPtr->atoms[n].y*bohr,bPtr->atoms[n].z*bohr);
  }

  std::vector<libint2::Atom> allAtoms;
  genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);

  arma::mat T1b,T2aaB,T2abB;
  double Ep,Em;
  std::vector<arma::mat> K1s,Ss;
  std::vector<std::vector<arma::mat> > V2s;
  bool useOld=true;
  if(bPtr->oldE == 0.0)
  {
    useOld=false;
  }
  bool fail;
  runPara(allAtoms,(*bPtr).basis,(*bPtr).nAOs,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).prodTable,Ep,bPtr->T1,bPtr->T2aa,bPtr->T2ab,K1s,V2s,bPtr->Cs,bPtr->nOcc,bPtr->nCore,true,true,bPtr->core,bPtr->numPe,useOld,bPtr->punchName,fail);

  //useOld=false;
  useOld=true;

  E=Ep;
  (*bPtr).E=E;

  double dx=0.000001;
  double *var;

  for(int m=0;m<(*bPtr).modes.size();m++)
  {
    switch((*bPtr).modes[m][1])
    {
      case 0:
	var = &(*bPtr).atoms[(*bPtr).modes[m][0]].x;
	break;
      case 1:
	var = &(*bPtr).atoms[(*bPtr).modes[m][0]].y;
	break;
      case 2:
	var = &(*bPtr).atoms[(*bPtr).modes[m][0]].z;
	break;
    }

    *var += dx;
    genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);
    std::vector<arma::mat> Cp=bPtr->Cs;
    std::vector<arma::mat> Cm=bPtr->Cs;
    T1b=bPtr->T1; T2aaB=bPtr->T2aa; T2abB=bPtr->T2ab;
    runPara(allAtoms,(*bPtr).basis,(*bPtr).nAOs,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).prodTable,Ep,T1b,T2aaB,T2abB,K1s,V2s,Cp,bPtr->nOcc,bPtr->nCore,false,false,bPtr->core,num,useOld,bPtr->punchName,fail);
	
    *var -= 2.0*dx;
    genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);
    T1b=bPtr->T1; T2aaB=bPtr->T2aa; T2abB=bPtr->T2ab;
    runPara(allAtoms,(*bPtr).basis,(*bPtr).nAOs,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).prodTable,Em,T1b,T2aaB,T2abB,K1s,V2s,Cm,bPtr->nOcc,bPtr->nCore,false,false,bPtr->core,num,useOld,bPtr->punchName,fail);
    //runPara(allAtoms,(*bPtr).basis,(*bPtr).nAOs,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).prodTable,Em,T1,T2aa,T2ab,K1s,V2s,Cm,nOcc,nCore,true,true,bPtr->core);

    grad[m] = (Ep-Em)/2.0/dx;
    bPtr->g[m]=grad[m];
    *var += dx;

/*Cs[0].print();
std::cout << std::endl;
Cp[0].print();
std::cout << std::endl;
Cm[0].print();
std::cout << std::endl;
Cp[0] -= Cm[0];
Cp[0].print();
std::cout << "\n\n\n";
*/

//std::cout << m << '\t' << grad[m] << std::endl;
  }

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
/*

  std::vector<libint2::Atom> allAtoms;
  genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);
  for(int a=0;a<allAtoms.size();a++)
  {
//    std::cout << allAtoms[a].x << '\t';
//    std::cout << allAtoms[a].y << '\t';
//    std::cout << allAtoms[a].z << '\n';
  }


  arma::mat T1,T2aa,T2ab;
  std::vector<arma::mat> Cs,K1s;
  std::vector<std::vector<arma::mat> > V2s;
  std::vector<int> nOcc;
  double Ep;
  runPara(allAtoms,(*bPtr).basis,(*bPtr).nAOs,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).prodTable,Ep,T1,T2aa,T2ab,K1s,V2s,Cs,nOcc,true,false);

  E=Ep;
  (*bPtr).E=E;

  arma::mat D1,D2aa,D2ab;
  makeD2(T1,T2aa,T2ab,D1,D2aa,D2ab);

//  std::cout << 2.0*trace(K1*D1)+trace(V2aa*D2aa)+trace(V2ab*D2ab) << std::endl;  
  //std::cout << std::endl;

  int nels=T1.n_rows;
  int virt=T1.n_cols;
  int nact=nels+virt;
  std::vector<double> eps(nact,0.0);

  for(int i=0;i<nact;i++)
  {
    eps[i] += K1(i,i);
    for(int j=0;j<nels;j++)
    {
      eps[i] += 2.0*V2ab(i*nact+j,i*nact+j)-V2ab(j*nact+i,i*nact+j);
    }
  }

  double EnucP,EnucM;
  std::vector<arma::mat> Sp,CsP,K1p,Sm,CsM,K1m;
  std::vector<std::vector<arma::mat> > V2p,V2m;

  double dx=0.000001;

  arma::mat K1b,V2b,V2bb;
  arma::mat K1t(nact,nact);
  arma::mat V2aaT(V2aa.n_rows,V2aa.n_cols);
  arma::mat V2abT(V2ab.n_rows,V2ab.n_cols);
  
  arma::mat U(nact,nact);
  //genInts(Enuc,atoms,basis,Ss,K1s,V2s,ao2mo,bas,nAOs,nels,prodTable);
  for(int a=1;a<(*bPtr).atoms.size();a++)
  {
    if(fabs((*bPtr).atoms[a].x) > 0.01)
    {
      (*bPtr).atoms[a].x += dx;
      genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);
      genInts(EnucP,allAtoms,(*bPtr).basis,Sp,K1p,V2p,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).nAOs,nels,(*bPtr).prodTable);

      (*bPtr).atoms[a].x -= 2.0*dx;
      genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);
      genInts(EnucM,allAtoms,(*bPtr).basis,Sm,K1m,V2m,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).nAOs,nels,(*bPtr).prodTable);

      EnucP = (EnucP-EnucM)/2.0/dx;      
      K1p[0] = (K1p[0]-K1m[0])/2.0/dx;
      Sp[0] = (Sp[0]-Sm[0])/2.0/dx;
      V2p[0][0] = (V2p[0][0]-V2m[0][0])/2.0/dx;

      rotateInts(K1p,V2p,Cs,(*bPtr).bas,K1b,V2b,V2bb,nOcc);
      Sp[0] = Cs[0].t()*Sp[0]*Cs[0];


      solveU(eps,K1b,V2b,Sp[0],V2ab,U,nels/2);
      makeKt(U,K1t,V2aaT,V2abT,K1,V2aa,V2ab);
      grad[3*a-3]=EnucP;
      grad[3*a-3]+=2.0*trace(K1b*D1)+trace(V2bb*D2aa)+trace(V2b*D2ab);
      grad[3*a-3]+=2.0*trace(K1t*D1)+trace(V2aaT*D2aa)+trace(V2abT*D2ab);

      (*bPtr).atoms[a].x += dx;
    }
    else
    {
      grad[3*a-3]=0.0;
    }
    if(fabs((*bPtr).atoms[a].y) > 0.01)
    {
      (*bPtr).atoms[a].y += dx;
      genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);
      genInts(EnucP,allAtoms,(*bPtr).basis,Sp,K1p,V2p,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).nAOs,nels,(*bPtr).prodTable);

      (*bPtr).atoms[a].y -= 2.0*dx;
      genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);
      genInts(EnucM,allAtoms,(*bPtr).basis,Sm,K1m,V2m,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).nAOs,nels,(*bPtr).prodTable);

      EnucP = (EnucP-EnucM)/2.0/dx;      
      K1p[0] = (K1p[0]-K1m[0])/2.0/dx;
      Sp[0] = (Sp[0]-Sm[0])/2.0/dx;
      V2p[0][0] = (V2p[0][0]-V2m[0][0])/2.0/dx;

      rotateInts(K1p,V2p,Cs,(*bPtr).bas,K1b,V2b,V2bb,nOcc);
      Sp[0] = Cs[0].t()*Sp[0]*Cs[0];


      solveU(eps,K1b,V2b,Sp[0],V2ab,U,nels/2);
      makeKt(U,K1t,V2aaT,V2abT,K1,V2aa,V2ab);
      grad[3*a-2]=EnucP;
      grad[3*a-2]+=2.0*trace(K1b*D1)+trace(V2bb*D2aa)+trace(V2b*D2ab);
      grad[3*a-2]+=2.0*trace(K1t*D1)+trace(V2aaT*D2aa)+trace(V2abT*D2ab);

      (*bPtr).atoms[a].y += dx;
    }
    else
    {
      grad[3*a-2]=0.0;
    }
    if(fabs((*bPtr).atoms[a].z) > 0.01)
    {
      (*bPtr).atoms[a].z += dx;
      genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);
      genInts(EnucP,allAtoms,(*bPtr).basis,Sp,K1p,V2p,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).nAOs,nels,(*bPtr).prodTable);

      (*bPtr).atoms[a].z -= 2.0*dx;
      genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);
      genInts(EnucM,allAtoms,(*bPtr).basis,Sm,K1m,V2m,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).nAOs,nels,(*bPtr).prodTable);

      EnucP = (EnucP-EnucM)/2.0/dx;      
      K1p[0] = (K1p[0]-K1m[0])/2.0/dx;
      Sp[0] = (Sp[0]-Sm[0])/2.0/dx;
      V2p[0][0] = (V2p[0][0]-V2m[0][0])/2.0/dx;

      rotateInts(K1p,V2p,Cs,(*bPtr).bas,K1b,V2b,V2bb,nOcc);
      Sp[0] = Cs[0].t()*Sp[0]*Cs[0];


      solveU(eps,K1b,V2b,Sp[0],V2ab,U,nels/2);
      makeKt(U,K1t,V2aaT,V2abT,K1,V2aa,V2ab);
      grad[3*a-1]=EnucP;
      grad[3*a-1]+=2.0*trace(K1b*D1)+trace(V2bb*D2aa)+trace(V2b*D2ab);
      grad[3*a-1]+=2.0*trace(K1t*D1)+trace(V2aaT*D2aa)+trace(V2abT*D2ab);

      (*bPtr).atoms[a].z += dx;
    }
    else
    {
      grad[3*a-1]=0.0;
    }
  }

  for(int a=1;a<(*bPtr).atoms.size();a++)
  {
//    std::cout << grad[3*a-3] << '\t' << grad[3*a-2] << '\t' << grad[3*a-1] << std::endl;
  }
*/
  return;
}

void solveU(std::vector<double> eps,const arma::mat &dK1,const arma::mat &dV2ab,const arma::mat &dS,const arma::mat &V2ab,arma::mat &U,int nels)
{
  int nMO=dK1.n_rows;
  int nAO=dS.n_rows;

  arma::mat dF=dK1;
  for(int i=0;i<nMO;i++)
  {
    for(int j=0;j<nMO;j++)
    {
      for(int k=0;k<nels;k++)
      {
	dF(i,j) += 2.0*dV2ab(i*nMO+k,j*nMO+k)-dV2ab(i*nMO+k,k*nMO+j);
      }
    }
  }
  
  arma::mat B=-dF;
  double x;
  for(int i=0;i<nMO;i++)
  {
    for(int j=0;j<nMO;j++)
    {
      B(i,j) += dS(i,j)*eps[j];
      x=0.0;
      for(int k=0;k<nels;k++)
      {
	for(int l=0;l<nels;l++)
	{
	  x += dS(k,l)*(2.0*V2ab(i*nMO+k,j*nMO+l)-V2ab(i*nMO+l,k*nMO+j));
	}
      }
      B(i,j) += x;
    }
  }

  arma::mat A(nMO*nMO,nMO*nMO,arma::fill::zeros);
  for(int i=0;i<nMO;i++)
  {
    for(int j=0;j<nMO;j++)
    {
      for(int k=0;k<nMO;k++)
      {
	for(int l=0;l<nMO;l++)
	{
	  A(i*nMO+k,j*nMO+l) += 4.0*V2ab(i*nMO+k,j*nMO+l);
	  A(i*nMO+k,j*nMO+l) -= V2ab(i*nMO+l,k*nMO+j);
	  A(i*nMO+k,j*nMO+l) -= V2ab(i*nMO+k,l*nMO+j);
	}
      }
    }
  }

  for(int i=0;i<nels;i++)
  {
    for(int j=nels;j<nMO;j++)
    {
      U(i,j)=B(i,j)/(eps[i]-eps[j]);
      U(j,i)=B(j,i)/(eps[j]-eps[i]);
    }
  }

  arma::mat Uold=U;
  U.zeros();
  arma::mat dU=Uold;
  while(fabs(dU.max())>pow(10,-6) || fabs(dU.min())>pow(10,-6))
  {
    Uold=U;
    for(int i=0;i<nels;i++)
    {
      for(int j=nels;j<nMO;j++)
      {
	U(i,j)=B(i,j);
	U(j,i)=B(j,i);
	x=0.0;
	for(int k=0;k<nels;k++)
	{
	  for(int l=nels;l<nMO;l++)
	  {
	    x += A(i*nMO+k,j*nMO+l)*Uold(l,k);
	  }
	}
	U(i,j) -= x;
	U(j,i) -= x;
	U(i,j) /= (eps[i]-eps[j]);
	U(j,i) /= (eps[j]-eps[i]);
      }
    }
    dU=U-Uold;
  }

  for(int i=0;i<nels;i++)
  {
    for(int j=i+1;j<nels;j++)
    {
      U(i,j)=B(i,j);
      U(j,i)=B(j,i);
      for(int k=0;k<nels;k++)
      {
	for(int l=nels;l<nMO;l++)
	{
	  U(i,j) -= A(i*nMO+k,j*nMO+l)*U(l,k);
	  U(j,i) -= A(i*nMO+k,j*nMO+l)*U(l,k);
	}
      }
      U(i,j) /= (eps[i]-eps[j]);
      U(j,i) /= (eps[j]-eps[i]);
    }
  }

  for(int i=nels;i<nMO;i++)
  {
    for(int j=i+1;j<nMO;j++)
    {
      U(i,j)=B(i,j);
      U(j,i)=B(j,i);
      for(int k=0;k<nels;k++)
      {
	for(int l=nels;l<nMO;l++)
	{
	  U(i,j) -= A(i*nMO+k,j*nMO+l)*U(l,k);
	  U(j,i) -= A(i*nMO+k,j*nMO+l)*U(l,k);
	}
      }
      U(i,j) /= (eps[i]-eps[j]);
      U(j,i) /= (eps[j]-eps[i]);
    }
  }

  for(int i=0;i<nMO;i++)
  {
    U(i,i) = -dS(i,i)/2.0;
  }

  return;
}

void makeKt(const arma::mat &U,arma::mat &K1t,arma::mat &V2aaT,arma::mat &V2abT,const arma::mat &K1,const arma::mat &V2aa,const arma::mat &V2ab)
{
  int nMO=K1.n_rows;
  K1t.zeros();
  V2aaT.zeros();
  V2abT.zeros();

  double x;
  for(int i=0;i<nMO;i++)
  {
    for(int j=0;j<nMO;j++)
    {
      for(int m=0;m<nMO;m++)
      {
	K1t(i,j)+=U(m,i)*K1(m,j)+U(m,j)*K1(i,m);
      }
    }
  }
  for(int i=0;i<nMO;i++)
  {
    for(int j=0;j<nMO;j++)
    {
      for(int k=0;k<nMO;k++)
      {
	for(int l=0;l<nMO;l++)
	{
	  x=0.0;
	  for(int m=0;m<nMO;m++)
	  {
	    x += U(m,i)*V2ab(m*nMO+j,k*nMO+l);
	    x += U(m,j)*V2ab(i*nMO+m,k*nMO+l);
	    x += U(m,k)*V2ab(i*nMO+j,m*nMO+l);
	    x += U(m,l)*V2ab(i*nMO+j,k*nMO+m);
	  }
	  V2abT(i*nMO+j,k*nMO+l) += x;
	}
      }
    }
  }
  for(int i=0;i<nMO;i++)
  {
    for(int j=i+1;j<nMO;j++)
    {
      for(int k=0;k<nMO;k++)
      {
	for(int l=k+1;l<nMO;l++)
	{
	  V2aaT(indexD(i,j,nMO),indexD(k,l,nMO))=2.0*(V2abT(i*nMO+j,k*nMO+l)-V2abT(i*nMO+j,l*nMO+k));
	}
      }
    }
  }

  return;
}

void gradEvalU(const real_1d_array &R,double &E,real_1d_array &grad,void *ptr)
{
std::cout << "evaluating" << std::endl;
  std::chrono::high_resolution_clock::time_point t1,t2;
  std::chrono::duration<double> time_span;

  setTime(t1,t1,time_span);
  int nT=0;
  int nT2;
  std::string label;
  
  double x;
  bundleNuc *bPtr=static_cast<bundleNuc*>(ptr);
  for(int m=0;m<bPtr->modes.size();m++)
  {
    switch(bPtr->modes[m][1])
    {
      case 0:
	bPtr->atoms[(*bPtr).modes[m][0]].x = R[m];
	break;
      case 1:
	bPtr->atoms[(*bPtr).modes[m][0]].y = R[m];
	break;
      case 2:
	bPtr->atoms[(*bPtr).modes[m][0]].z = R[m];
	break;
    }
  }

  std::vector<libint2::Atom> allAtoms;
  genAtoms(bPtr->basis,bPtr->pointGroup,bPtr->atoms,allAtoms);

  arma::mat T1,T2aa,T2ab;
  std::vector<arma::mat> Cs,K1s,Ss;
  std::vector<std::vector<arma::mat> > V2s;
  std::vector<int> nOcc;
  double Ep;
  //runPara(allAtoms,bPtr->basis,bPtr->nAOs,bPtr->ao2mo,bPtr->bas,bPtr->prodTable,Ep,T1,T2aa,T2ab,K1s,V2s,Cs,nOcc,true,false);

  E=Ep;
  bPtr->E=E;

  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "Single point";
  nT++; t1=t2;

  arma::mat D1,D2aa,D2ab;
  makeD2(T1,T2aa,T2ab,D1,D2aa,D2ab);

  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "make D2";
  nT++; t1=t2;

  std::vector<int> nMO(nOcc.size());
  for(int p=0;p<nOcc.size();p++)
  {
    nMO[p]=K1s[p].n_cols;
  }

  std::vector<arma::mat> D1s;
  std::vector<std::vector<arma::mat> > D2aaS,D2abS;
  //rotateForward(nOcc,D1,D2aa,D2ab,nMO,D1s,D2aaS,D2abS,bPtr->bas);

  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "make D2sym";
  nT++; t1=t2;

  std::vector<std::vector<arma::mat> > V2aaS;
  makeV2aa(V2s,V2aaS,bPtr->bas,nMO);

  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "make V2aa";
  nT++; t1=t2;

  int nels=T1.n_rows;
  int virt=T1.n_cols;
  int nact=nels+virt;

  std::vector<std::vector<double> > eps(nOcc.size());
  for(int p=0;p<nOcc.size();p++)
  {
    for(int i=0;i<K1s[p].n_cols;i++)
    {
      x=K1s[p](i,i);
      for(int q=0;q<nOcc.size();q++)
      {
	for(int j=0;j<nOcc[q];j++)
	{
	  x += 2.0*V2s[p*nOcc.size()+q][p*nOcc.size()+q](i*K1s[q].n_cols+j,i*K1s[q].n_cols+j);
	  x -= V2s[q*nOcc.size()+p][p*nOcc.size()+q](j*K1s[p].n_cols+i,i*K1s[q].n_cols+j);
	}
      }
      eps[p].push_back(x);
//std::cout << eps[p][eps[p].size()-1] << std::endl;
    }
  }
  double EnucP,EnucM;
  std::vector<arma::mat> Sp,CsP,K1p,Sm,CsM,K1m;
  std::vector<std::vector<arma::mat> > V2p,V2m;

  double dx=0.000001;
  double *var;

  arma::mat K1b,V2aaB,V2abB;
  arma::mat K1c,V2aaC,V2abC;

  std::vector<arma::mat> K1t;
  std::vector<std::vector<arma::mat> > V2t;
  
  std::vector<arma::mat> U(nOcc.size());

  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "last stuff";
  nT++; t1=t2;

  //for(int m=0;m<1;m++)
  for(int m=0;m<(*bPtr).modes.size();m++)
  {
    setTime(t1,t1,time_span);

    genIntsDer(EnucP,bPtr->atoms,bPtr->basis,Sp,K1p,V2p,bPtr->ao2mo,bPtr->bas,bPtr->nAOs,bPtr->prodTable,bPtr->pointGroup,bPtr->modes[m],dx);
    for(int r=0;r<K1s.size();r++)
    {
      Sp[r] = Cs[r].t()*Sp[r]*Cs[r];
    }

    setTime(t1,t2,time_span);
    label = "make deriv"; findLabel(label,bPtr->tLabel,nT);
    bPtr->tLabel[nT] = label;
    bPtr->ts[nT] += time_span;
    nT++; t1=t2;

    //rotateInts(K1p,V2p,Cs,bPtr->bas,K1b,V2abB,V2aaB,nOcc,nCore,Ecore);

    setTime(t1,t2,time_span);
    label = "rotate ints"; findLabel(label,bPtr->tLabel,nT);
    bPtr->tLabel[nT] = label;
    bPtr->ts[nT] += time_span;
    nT++; t1=t2;

    std::vector<arma::mat> B;
    solveUsym(eps,K1p,V2p,Sp,V2s,U,nOcc,B);

    for(int p=0;p<U.size();p++)
    {
//      U[p].print();
//    std::cout << std::endl;
    }

    setTime(t1,t2,time_span);
    label = "solve U"; findLabel(label,bPtr->tLabel,nT);
    bPtr->tLabel[nT] = label;
    bPtr->ts[nT] += time_span;
    nT++; t1=t2;

    makeKtsym(U,K1t,V2t,K1s,V2s);

    setTime(t1,t2,time_span);
    label = "make Kt"; findLabel(label,bPtr->tLabel,nT);
    bPtr->tLabel[nT] = label;
    bPtr->ts[nT] += time_span;
    nT++; t1=t2;

    rotateBack(K1t,V2t,nOcc,K1c,V2aaC,V2abC);

    setTime(t1,t2,time_span);
    label = "make Kt matrix"; findLabel(label,bPtr->tLabel,nT);
    bPtr->tLabel[nT] = label;
    bPtr->ts[nT] += time_span;
    nT++; t1=t2;

    grad[m]=EnucP;
    grad[m]+=2.0*trace(K1b*D1)+trace(V2aaB*D2aa)+trace(V2abB*D2ab);
    //grad[m]=0.0;
    grad[m]+=2.0*trace(K1c*D1)+trace(V2aaC*D2aa)+trace(V2abC*D2ab);
    bPtr->g[m]=grad[m];

    //std::cout << m << '\t' << grad[m] << std::endl;

    setTime(t1,t2,time_span);
    label = "final evaluation"; findLabel(label,bPtr->tLabel,nT);
    bPtr->tLabel[nT] = label;
    bPtr->ts[nT] += time_span;
    nT++; t1=t2;

  }

  return;
}

void solveUsym(std::vector<std::vector<double> > eps,const std::vector<arma::mat> &dK1,const std::vector<std::vector<arma::mat> > &dV2ab,const std::vector<arma::mat> &dS,const std::vector<std::vector<arma::mat> > &V2ab,std::vector<arma::mat> &U,std::vector<int> nOcc,std::vector<arma::mat> &B)
{
  std::vector<int> nMO;
  std::vector<int> nAO;
  
  for(int p=0;p<dK1.size();p++)
  {
    nMO.push_back(dK1[p].n_cols);
    nAO.push_back(dS[p].n_cols);
  }

  std::vector<arma::mat> dF=dK1;
  for(int p=0;p<nOcc.size();p++)
  {
    for(int i=0;i<nMO[p];i++)
    {
      for(int j=0;j<nMO[p];j++)
      {
	for(int q=0;q<nOcc.size();q++)
	{
	  for(int k=0;k<nOcc[q];k++)
	  {
	    dF[p](i,j) += 2.0*dV2ab[p*nOcc.size()+q][p*nOcc.size()+q](i*nMO[q]+k,j*nMO[q]+k);
	    dF[p](i,j) -= dV2ab[q*nOcc.size()+p][p*nOcc.size()+q](k*nMO[p]+i,j*nMO[q]+k);
	  }
	}
      }
    }
  }

  
  B=dF;
  //std::vector<arma::mat> B=dF;
  for(int p=0;p<B.size();p++)
  {
    B[p] *= -1.0;
  }
  double x;
  for(int p=0;p<B.size();p++)
  {
    for(int i=0;i<nMO[p];i++)
    {
      for(int j=0;j<nMO[p];j++)
      {
	B[p](i,j) += dS[p](i,j)*eps[p][j];
	x=0.0;
	for(int q=0;q<B.size();q++)
	{
	  for(int k=0;k<nOcc[q];k++)
	  {
	    for(int l=0;l<nOcc[q];l++)
	    {
	      x += 2.0*dS[q](k,l)*V2ab[p*B.size()+q][p*B.size()+q](i*nMO[q]+k,j*nMO[q]+l);
	      x -= dS[q](k,l)*V2ab[p*B.size()+p][q*B.size()+q](i*nMO[p]+j,k*nMO[q]+l);
	      //x -= dS[q](k,l)*V2ab[p*B.size()+q][q*B.size()+p](i*nMO[q]+l,k*nMO[p]+j);
	    }
	  }
	}
	B[p](i,j) += x;
      }
    }
  }

/*  for(int i=0;i<nMO;i++)
  {
    for(int j=0;j<nMO;j++)
    {
      B(i,j) += dS(i,j)*eps[j];
      x=0.0;
      for(int k=0;k<nels;k++)
      {
	for(int l=0;l<nels;l++)
	{
	  x += dS(k,l)*(2.0*V2ab(i*nMO+k,j*nMO+l)-V2ab(i*nMO+l,k*nMO+j));
	}
      }
      B(i,j) += x;
    }
  }
*/

  std::vector<std::vector<arma::mat> > A = V2ab;
  for(int p=0;p<V2ab.size();p++)
  {
    for(int q=0;q<V2ab[p].size();q++)
    {
      A[p][q] *= 4.0;
    }
  }
  for(int p=0;p<B.size();p++)
  {
    for(int q=0;q<B.size();q++)
    {
      for(int r=0;r<B.size();r++)
      {
	for(int s=0;s<B.size();s++)
	{
	  if(V2ab[p*B.size()+r][q*B.size()+s].n_rows >0 && V2ab[p*B.size()+r][q*B.size()+s].n_cols > 0)
	  {
	    for(int i=0;i<nMO[p];i++)
	    {
	      for(int j=0;j<nMO[q];j++)
	      {
		for(int k=0;k<nMO[r];k++)
		{
		  for(int l=0;l<nMO[s];l++)
		  {
		    A[p*B.size()+r][q*B.size()+s](i*nMO[r]+k,j*nMO[s]+l) -= V2ab[p*B.size()+s][r*B.size()+q](i*nMO[s]+l,k*nMO[q]+j);
		    A[p*B.size()+r][q*B.size()+s](i*nMO[r]+k,j*nMO[s]+l) -= V2ab[p*B.size()+r][s*B.size()+q](i*nMO[r]+k,l*nMO[q]+j);
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }


//  arma::mat A(nMO*nMO,nMO*nMO,arma::fill::zeros);
  
/*  for(int i=0;i<nMO;i++)
  {
    for(int j=0;j<nMO;j++)
    {
      for(int k=0;k<nMO;k++)
      {
	for(int l=0;l<nMO;l++)
	{
	  A(i*nMO+k,j*nMO+l) += 4.0*V2ab(i*nMO+k,j*nMO+l);
	  A(i*nMO+k,j*nMO+l) -= V2ab(i*nMO+l,k*nMO+j);
	  A(i*nMO+k,j*nMO+l) -= V2ab(i*nMO+k,l*nMO+j);
	}
      }
    }
  }
*/

  for(int p=0;p<B.size();p++)
  {
    U[p].set_size(nMO[p],nMO[p]);
    U[p].zeros();
    for(int i=0;i<nMO[p];i++)
    {
      for(int j=i+1;j<nMO[p];j++)
      {
	//U[p](i,j)=B[p](i,j)/(eps[p][i]-eps[p][j]);
	U[p](j,i)=B[p](j,i)/(eps[p][j]-eps[p][i]);
      }
    }
  }

/*  for(int i=0;i<nels;i++)
  {
    for(int j=nels;j<nMO;j++)
    {
      U(i,j)=B(i,j)/(eps[i]-eps[j]);
      U(j,i)=B(j,i)/(eps[j]-eps[i]);
    }
  }
*/

  std::vector<arma::mat> Uold=U;
  double dConv=0.0;
  for(int p=0;p<U.size();p++)
  {
//    U[p].zeros();
    dConv += arma::norm(Uold[p]);
  }
  //std::vector<arma::mat> dU=Uold;
  while(dConv > 0.000001)
  {
    Uold=U;
    dConv=0.0;
    for(int p=0;p<U.size();p++)
    {
      for(int i=0;i<nMO[p];i++)
      {
	for(int j=i+1;j<nMO[p];j++)
	{
	  U[p](i,j)=B[p](i,j);
	  U[p](j,i)=B[p](j,i);
	  x=0.0;
	  for(int q=0;q<U.size();q++)
	  {
	    for(int k=0;k<nOcc[q];k++)
	    {
	      for(int l=nOcc[q];l<nMO[q];l++)
	      {
		x += A[p*B.size()+q][p*B.size()+q](i*nMO[q]+k,j*nMO[q]+l)*Uold[q](l,k);
	      }
	    }
	  }
	  U[p](i,j) -= x;
	  U[p](j,i) -= x;
	  U[p](i,j) /= (eps[p][i]-eps[p][j]);
	  U[p](j,i) /= (eps[p][j]-eps[p][i]);
	}
      }
      dConv += arma::norm(U[p]-Uold[p]);
      //U[p].print();
      //std::cout << std::endl;
    }
    //std::cout << dConv << std::endl;

/*    for(int i=0;i<nels;i++)
    {
      for(int j=nels;j<nMO;j++)
      {
	U(i,j)=B(i,j);
	U(j,i)=B(j,i);
	x=0.0;
	for(int k=0;k<nels;k++)
	{
	  for(int l=nels;l<nMO;l++)
	  {
	    x += A(i*nMO+k,j*nMO+l)*Uold(l,k);
	  }
	}
	U(i,j) -= x;
	U(j,i) -= x;
	U(i,j) /= (eps[i]-eps[j]);
	U(j,i) /= (eps[j]-eps[i]);
      }
    }
    dU=U-Uold;
*/
  }

/*  for(int p=0;p<U.size();p++)
  {
    for(int i=0;i<nOcc[p];i++)
    {
      for(int j=i+1;j<nOcc[p];j++)
      {
	U[p](i,j)=B[p](i,j);
	U[p](j,i)=B[p](j,i);
	for(int q=0;q<U.size();q++)
	{
	  for(int k=0;k<nOcc[q];k++)
	  {
	    for(int l=nOcc[q];l<nMO[q];l++)
	    {
	      U[p](i,j) -= A[p*B.size()+q][p*B.size()+q](i*nMO[q]+k,j*nMO[q]+l)*U[q](l,k);
	      U[p](j,i) -= A[p*B.size()+q][p*B.size()+q](i*nMO[q]+k,j*nMO[q]+l)*U[q](l,k);
	    }
	  }
	}
	U[p](i,j) /= (eps[p][i]-eps[p][j]);
	U[p](j,i) /= (eps[p][j]-eps[p][i]);
      }
    }
*/
  for(int p=0;p<U.size();p++)
  {
    for(int i=nOcc[p];i<nMO[p];i++)
    {
      for(int j=i+1;j<nMO[p];j++)
      {
	U[p](i,j)=B[p](i,j);
	U[p](j,i)=B[p](j,i);
	for(int q=0;q<U.size();q++)
	{
	  for(int k=0;k<nOcc[q];k++)
	  {
	    for(int l=nOcc[q];l<nMO[q];l++)
	    {
	      U[p](i,j) -= A[p*B.size()+q][p*B.size()+q](i*nMO[q]+k,j*nMO[q]+l)*U[q](l,k);
	      U[p](j,i) -= A[p*B.size()+q][p*B.size()+q](i*nMO[q]+k,j*nMO[q]+l)*U[q](l,k);
	    }
	  }
	}
	U[p](i,j) /= (eps[p][i]-eps[p][j]);
	U[p](j,i) /= (eps[p][j]-eps[p][i]);
      }
    }
  }
/**/
  for(int p=0;p<U.size();p++)
  {
    for(int i=0;i<nMO[p];i++)
    {
      U[p](i,i) -= dS[p](i,i)/2.0;
    }
  }

/*  for(int i=0;i<nels;i++)
  {
    for(int j=i+1;j<nels;j++)
    {
      U(i,j)=B(i,j);
      U(j,i)=B(j,i);
      for(int k=0;k<nels;k++)
      {
	for(int l=nels;l<nMO;l++)
	{
	  U(i,j) -= A(i*nMO+k,j*nMO+l)*U(l,k);
	  U(j,i) -= A(i*nMO+k,j*nMO+l)*U(l,k);
	}
      }
      U(i,j) /= (eps[i]-eps[j]);
      U(j,i) /= (eps[j]-eps[i]);
    }
  }

  for(int i=nels;i<nMO;i++)
  {
    for(int j=i+1;j<nMO;j++)
    {
      U(i,j)=B(i,j);
      U(j,i)=B(j,i);
      for(int k=0;k<nels;k++)
      {
	for(int l=nels;l<nMO;l++)
	{
	  U(i,j) -= A(i*nMO+k,j*nMO+l)*U(l,k);
	  U(j,i) -= A(i*nMO+k,j*nMO+l)*U(l,k);
	}
      }
      U(i,j) /= (eps[i]-eps[j]);
      U(j,i) /= (eps[j]-eps[i]);
    }
  }

  for(int i=0;i<nMO;i++)
  {
    U(i,i) = -dS(i,i)/2.0;
  }
*/
  for(int p=0;p<U.size();p++)
  {
    for(int i=0;i<nOcc[p];i++)
    {
      for(int j=nOcc[p];j<nMO[p];j++)
      {
//	U[p](i,j) = -dS[p](i,j)-U[p](j,i);
      }
    }
  }

  return;
}

void makeKtsym(const std::vector<arma::mat> &U,std::vector<arma::mat> &K1t,std::vector<std::vector<arma::mat> > &V2t,const std::vector<arma::mat> &K1s,const std::vector<std::vector<arma::mat> > &V2s)
{
  std::vector<int> nMO;
  K1t.resize(K1s.size());
  V2t.resize(V2s.size());
  for(int p=0;p<K1s.size();p++)
  {
    nMO.push_back(K1s[p].n_cols);
    K1t[p] = arma::mat(nMO[p],nMO[p],arma::fill::zeros);
  }
  for(int p=0;p<V2s.size();p++)
  {
    V2t[p].resize(V2s[p].size());
    for(int q=0;q<V2s[p].size();q++)
    {
      V2t[p][q] = arma::mat(V2s[p][q].n_rows,V2s[p][q].n_cols,arma::fill::zeros);
    }
  }
  //int nMO=K1.n_rows;
  //K1t.zeros();
  //V2aaT.zeros();
  //V2abT.zeros();

  double x;
  for(int p=0;p<K1t.size();p++)
  {
    for(int i=0;i<nMO[p];i++)
    {
      for(int j=0;j<nMO[p];j++)
      {
	for(int m=0;m<nMO[p];m++)
	{
	  K1t[p](i,j) += U[p](m,i)*K1s[p](m,j)+U[p](m,j)*K1s[p](i,m); 
        }
      }
    }
  }
/*  for(int i=0;i<nMO;i++)
  {
    for(int j=0;j<nMO;j++)
    {
      for(int m=0;m<nMO;m++)
      {
	K1t(i,j)+=U(m,i)*K1(m,j)+U(m,j)*K1(i,m);
      }
    }
  }
*/
  for(int p=0;p<K1s.size();p++)
  {
    for(int q=0;q<K1s.size();q++)
    {
      for(int r=0;r<K1s.size();r++)
      {
	for(int s=0;s<K1s.size();s++)
	{
	  if(V2s[p*K1s.size()+q][r*K1s.size()+s].n_rows>0 && V2s[p*K1s.size()+q][r*K1s.size()+s].n_cols>0)
	  {
	    for(int i=0;i<nMO[p];i++)
	    {
	      for(int j=0;j<nMO[q];j++)
	      {
		for(int k=0;k<nMO[r];k++)
		{
		  for(int l=0;l<nMO[s];l++)
		  {
		    x=0.0;
		    for(int m=0;m<nMO[p];m++)
		    {
		      x += U[p](m,i)*V2s[p*K1s.size()+q][r*K1s.size()+s](m*nMO[q]+j,k*nMO[s]+l);
		    }
		    for(int m=0;m<nMO[q];m++)
		    {
		      x += U[q](m,j)*V2s[p*K1s.size()+q][r*K1s.size()+s](i*nMO[q]+m,k*nMO[s]+l);
		    }
		    for(int m=0;m<nMO[r];m++)
		    {
		      x += U[r](m,k)*V2s[p*K1s.size()+q][r*K1s.size()+s](i*nMO[q]+j,m*nMO[s]+l);
		    }
		    for(int m=0;m<nMO[s];m++)
		    {
		      x += U[s](m,l)*V2s[p*K1s.size()+q][r*K1s.size()+s](i*nMO[q]+j,k*nMO[s]+m);
		    }
		    V2t[p*K1s.size()+q][r*K1s.size()+s](i*nMO[q]+j,k*nMO[s]+l) += x;
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

/*  for(int i=0;i<nMO;i++)
  {
    for(int j=0;j<nMO;j++)
    {
      for(int k=0;k<nMO;k++)
      {
	for(int l=0;l<nMO;l++)
	{
	  x=0.0;
	  for(int m=0;m<nMO;m++)
	  {
	    x += U(m,i)*V2ab(m*nMO+j,k*nMO+l);
	    x += U(m,j)*V2ab(i*nMO+m,k*nMO+l);
	    x += U(m,k)*V2ab(i*nMO+j,m*nMO+l);
	    x += U(m,l)*V2ab(i*nMO+j,k*nMO+m);
	  }
	  V2abT(i*nMO+j,k*nMO+l) += x;
	}
      }
    }
  }
*/
  return;
}

void rotateBack(const std::vector<arma::mat> &K1s,const std::vector<std::vector<arma::mat> > &V2s,std::vector<int> nOcc,arma::mat &K1,arma::mat &V2aa,arma::mat &V2ab)
{
  std::vector<int> bas=nOcc;

  int nels=0;
  for(int i=0;i<nOcc.size();i++)
  {
    nels += nOcc[i];
  }
  int nMO=0;
  for(int r=0;r<K1s.size();r++)
  {
    nMO += K1s[r].n_cols;
  }

  std::vector<int> nVirt;
  for(int i=0;i<nOcc.size();i++)
  {
    nVirt.push_back(K1s[i].n_cols-nOcc[i]);
  }

  K1.set_size(nMO,nMO);
  K1.zeros();

  int row1,row2,col1,col2;
  int nRow=0; int nCol=0;
  for(int p=0;p<K1s.size();p++)
  {
    for(int n1=0;n1<nOcc[p];n1++)
    {
      for(int n2=0;n2<nOcc[p];n2++)
      {
	K1(nRow+n2,nRow+n1) = K1s[p](n2,n1);
      }
      for(int n2=0;n2<nVirt[p];n2++)
      {
	K1(nCol+n2+nels,nRow+n1) = K1s[p](n2+nOcc[p],n1);
      }
    }
    for(int n1=0;n1<nVirt[p];n1++)
    {
      for(int n2=0;n2<nOcc[p];n2++)
      {
	K1(nRow+n2,nCol+n1+nels) = K1s[p](n2,n1+nOcc[p]);
      }
      for(int n2=0;n2<nVirt[p];n2++)
      {
	K1(nCol+n2+nels,nCol+n1+nels) = K1s[p](n2+nOcc[p],n1+nOcc[p]);
      }
      
    }

    nRow += nOcc[p];
    nCol += nVirt[p];
  }

  int o1,o2,o3,o4;
  int v1,v2,v3,v4;

  V2ab.set_size(nMO*nMO,nMO*nMO);
  V2ab.zeros();

  o1=0; v1=nels;
  for(int p=0;p<K1s.size();p++)
  {
    o2=0; v2=nels;
    for(int q=0;q<K1s.size();q++)
    {
      o3=0; v3=nels;
      for(int r=0;r<K1s.size();r++)
      {
	o4=0; v4=nels;
	for(int s=0;s<K1s.size();s++)
	{
	  if(V2s[r*bas.size()+s][p*bas.size()+q].n_rows > 0 && V2s[r*bas.size()+s][p*bas.size()+q].n_cols > 0)
	  {
	    for(int n1=0;n1<nOcc[p];n1++)
	    {
	      for(int n2=0;n2<nOcc[q];n2++)
	      {
		for(int n3=0;n3<nOcc[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+o4+n4,(o1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q](n3*K1s[s].n_cols+n4,n1*K1s[q].n_cols+n2);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+v4+n4,(o1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q](n3*K1s[s].n_cols+n4+nOcc[s],n1*K1s[q].n_cols+n2);
		  }
		}
		for(int n3=0;n3<nVirt[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+o4+n4,(o1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r])*K1s[s].n_cols+n4,n1*K1s[q].n_cols+n2);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+v4+n4,(o1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r])*K1s[s].n_cols+n4+nOcc[s],n1*K1s[q].n_cols+n2);
		  }
		}
	      }
	      for(int n2=0;n2<nVirt[q];n2++)
	      {
		for(int n3=0;n3<nOcc[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+o4+n4,(o1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q](n3*K1s[s].n_cols+n4,n1*K1s[q].n_cols+n2+nOcc[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+v4+n4,(o1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q](n3*K1s[s].n_cols+n4+nOcc[s],n1*K1s[q].n_cols+n2+nOcc[q]);
		  }
		}
		for(int n3=0;n3<nVirt[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+o4+n4,(o1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r])*K1s[s].n_cols+n4,n1*K1s[q].n_cols+n2+nOcc[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+v4+n4,(o1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r])*K1s[s].n_cols+n4+nOcc[s],n1*K1s[q].n_cols+n2+nOcc[q]);
		  }
		}
	      }
	    }
	    for(int n1=0;n1<nVirt[p];n1++)
	    {
	      for(int n2=0;n2<nOcc[q];n2++)
	      {
		for(int n3=0;n3<nOcc[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+o4+n4,(v1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q](n3*K1s[s].n_cols+n4,(n1+nOcc[p])*K1s[q].n_cols+n2);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+v4+n4,(v1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q](n3*K1s[s].n_cols+n4+nOcc[s],(n1+nOcc[p])*K1s[q].n_cols+n2);
		  }
		}
		for(int n3=0;n3<nVirt[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+o4+n4,(v1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r])*K1s[s].n_cols+n4,(n1+nOcc[p])*K1s[q].n_cols+n2);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+v4+n4,(v1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r])*K1s[s].n_cols+n4+nOcc[s],(n1+nOcc[p])*K1s[q].n_cols+n2);
		  }
		}
	      }
	      for(int n2=0;n2<nVirt[q];n2++)
	      {
		for(int n3=0;n3<nOcc[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+o4+n4,(v1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q](n3*K1s[s].n_cols+n4,(n1+nOcc[p])*K1s[q].n_cols+n2+nOcc[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+v4+n4,(v1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q](n3*K1s[s].n_cols+n4+nOcc[s],(n1+nOcc[p])*K1s[q].n_cols+n2+nOcc[q]);
		  }
		}
		for(int n3=0;n3<nVirt[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+o4+n4,(v1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r])*K1s[s].n_cols+n4,(n1+nOcc[p])*K1s[q].n_cols+n2+nOcc[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+v4+n4,(v1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r])*K1s[s].n_cols+n4+nOcc[s],(n1+nOcc[p])*K1s[q].n_cols+n2+nOcc[q]);
		  }
		}
	      }
	    }
	  }

	  o4 += nOcc[s];
	  v4 += nVirt[s];
	}
	o3 += nOcc[r];
	v3 += nVirt[r];
      }
      o2 += nOcc[q];
      v2 += nVirt[q];
    }
    o1 += nOcc[p];
    v1 += nVirt[p];
  }

  V2aa.set_size(nMO*(nMO-1)/2,nMO*(nMO-1)/2);
  V2aa.zeros();

  for(int i=0;i<nMO;i++)
  {
    for(int j=i+1;j<nMO;j++)
    {
      for(int k=0;k<nMO;k++)
      {
	for(int l=k+1;l<nMO;l++)
	{
	  V2aa(indexD(k,l,nMO),indexD(i,j,nMO)) = 2.0*(V2ab(k*nMO+l,i*nMO+j)-V2ab(l*nMO+k,i*nMO+j));
	}
      }
    }
  }

  return;
}

void rotateForward(const std::vector<int> &nOcc,const arma::mat &D1,arma::mat &D2aa,const arma::mat &D2ab,const std::vector<int> &nMOs,std::vector<arma::mat> &D1s,std::vector<std::vector<arma::mat> > &D2aaS,std::vector<std::vector<arma::mat> > &D2abS,const std::vector<std::vector<std::vector<int> > > &bas,const std::vector<int> &nc)
{
  int nels=0;
  int nMO=D1.n_rows;
  std::vector<int> nVirt(nOcc.size());
  for(int p=0;p<nOcc.size();p++)
  {
    nels += nOcc[p];
//    nMO += nMOs[p];
    nVirt[p]=nMOs[p]-nOcc[p]-nc[p];
  }

  D1s.resize(nOcc.size());
  for(int p=0;p<D1s.size();p++)
  {
    D1s[p].set_size(nMOs[p],nMOs[p]);
    D1s[p].zeros();
  }

  int o1,o2,o3,o4;
  int v1,v2,v3,v4;
  o1=0;
  v1=0;
  for(int p=0;p<D1s.size();p++)
  {
    for(int i=0;i<nc[p];i++)
    {
      D1s[p](i,i)=1.0;
    }
    for(int i=0;i<nOcc[p];i++)
    {
      for(int j=0;j<nOcc[p];j++)
      {
	D1s[p](i+nc[p],j+nc[p]) += D1(i+o1,j+o1);
      }
      for(int j=0;j<nVirt[p];j++)
      {
	D1s[p](i+nc[p],j+nOcc[p]+nc[p]) += D1(i+o1,j+v1+nels);
      }
    }
    for(int i=0;i<nVirt[p];i++)
    {
      for(int j=0;j<nOcc[p];j++)
      {
	D1s[p](i+nOcc[p]+nc[p],j+nc[p]) += D1(i+v1+nels,j+o1);
      }
      for(int j=0;j<nVirt[p];j++)
      {
	D1s[p](i+nOcc[p]+nc[p],j+nOcc[p]+nc[p]) += D1(i+v1+nels,j+v1+nels);
      }
    }
    o1 += nOcc[p];
    v1 += nVirt[p];
  }

  D2abS.resize(bas.size()*bas.size());
  D2aaS.resize(bas.size()*bas.size());
  for(int p=0;p<D2abS.size();p++)
  {
    D2abS[p].resize(bas.size()*bas.size());
    D2aaS[p].resize(bas.size()*bas.size());
  }
  for(int t=0;t<bas.size();t++)
  {
    for(int a=0;a<bas[t].size();a++)
    {
      int p=bas[t][a][0];
      int q=bas[t][a][1];
      for(int b=0;b<bas[t].size();b++)
      {
	int r=bas[t][b][0];
	int s=bas[t][b][1];
	D2abS[p*bas.size()+q][r*bas.size()+s].set_size(nMOs[p]*nMOs[q],nMOs[r]*nMOs[s]);
	D2abS[p*bas.size()+q][r*bas.size()+s].zeros();
      }
    }
  }

  for(int p=0;p<bas.size();p++)
  {
    for(int q=0;q<bas.size();q++)
    {
      D2aaS[p*bas.size()+p][q*bas.size()+q].set_size(nMOs[p]*(nMOs[p]-1)/2,nMOs[q]*(nMOs[q]-1)/2);
      D2aaS[p*bas.size()+p][q*bas.size()+q].zeros();
    }
    for(int q=p+1;q<bas.size();q++)
    {
      for(int r=0;r<bas.size();r++)
      {
	for(int s=r+1;s<bas.size();s++)
	{
	  D2aaS[p*bas.size()+q][r*bas.size()+s].set_size(nMOs[p]*nMOs[q],nMOs[r]*nMOs[s]);
	  D2aaS[p*bas.size()+q][r*bas.size()+s].zeros();
	}
      }
    }
  }

  for(int p=0;p<bas.size();p++)
  {
    for(int q=0;q<bas.size();q++)
    {
      for(int i=0;i<nc[p];i++)
      {
	for(int j=0;j<nc[q];j++)
	{
	  D2abS[p*bas.size()+q][p*bas.size()+q](i*nMOs[q]+j,i*nMOs[q]+j)+=1.0;
	}
	for(int j=nc[q];j<nMOs[q];j++)
	{
	  for(int k=nc[q];k<nMOs[q];k++)
	  {
	    D2abS[p*bas.size()+q][p*bas.size()+q](i*nMOs[q]+j,i*nMOs[q]+k)+=D1s[q](j,k);
	    D2abS[q*bas.size()+p][q*bas.size()+p](j*nMOs[p]+i,k*nMOs[p]+i)+=D1s[q](j,k);
	  }
	}
      }
    }
  }
  for(int p=0;p<bas.size();p++)
  {
    for(int i=0;i<nc[p];i++)
    {
      for(int j=i+1;j<nc[p];j++)
      {
	D2aaS[p*bas.size()+p][p*bas.size()+p](indexD(i,j,nMOs[p]),indexD(i,j,nMOs[p]))+=1.0;
      }
      for(int j=nc[p];j<nMOs[p];j++)
      {
	for(int k=nc[p];k<nMOs[p];k++)
	{
	  D2aaS[p*bas.size()+p][p*bas.size()+p](indexD(i,j,nMOs[p]),indexD(i,k,nMOs[p]))+=D1s[p](j,k);
	}
      }
    }
  }
  for(int p=0;p<bas.size();p++)
  {
    for(int q=p+1;q<bas.size();q++)
    {
      for(int i=0;i<nc[p];i++)
      {
	for(int j=0;j<nc[q];j++)
	{
	  D2aaS[p*bas.size()+q][p*bas.size()+q](i*nMOs[q]+j,i*nMOs[q]+j)+=1.0;
	}
	for(int j=nc[q];j<nMOs[q];j++)
	{
	  for(int k=nc[q];k<nMOs[q];k++)
	  {
	    D2aaS[p*bas.size()+q][p*bas.size()+q](i*nMOs[q]+j,i*nMOs[q]+k)+=D1s[q](j,k);
	  }
	}
      }
      for(int j=nc[p];j<nMOs[p];j++)
      {
	for(int k=nc[p];k<nMOs[p];k++)
	{
	  for(int i=0;i<nc[q];i++)
	  {
	    D2aaS[p*bas.size()+q][p*bas.size()+q](j*nMOs[q]+i,k*nMOs[q]+i)+=D1s[p](j,k);
	  }
	}
      }
    }
  } 

  o1=0; v1=0;
  for(int p=0;p<bas.size();p++)
  {
    o2=0; v2=0;
    for(int q=0;q<bas.size();q++)
    {
      o3=0; v3=0;
      for(int r=0;r<bas.size();r++)
      {
	o4=0; v4=0;
	for(int s=0;s<bas.size();s++)
	{
	  if(D2abS[p*bas.size()+q][r*bas.size()+s].n_rows>0 && D2abS[p*bas.size()+q][r*bas.size()+s].n_cols>0)
	  {
	    for(int k=0;k<nOcc[r];k++)
	    {
	      for(int l=0;l<nOcc[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nc[q],(k+nc[r])*nMOs[s]+l+nc[s])=D2ab((i+o1)*nMO+j+o2,(k+o3)*nMO+l+o4);
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nc[r])*nMOs[s]+l+nc[s])=D2ab((i+o1)*nMO+j+v2+nels,(k+o3)*nMO+l+o4);
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nc[q],(k+nc[r])*nMOs[s]+l+nc[s])=D2ab((i+v1+nels)*nMO+j+o2,(k+o3)*nMO+l+o4);
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nc[r])*nMOs[s]+l+nc[s])=D2ab((i+v1+nels)*nMO+j+v2+nels,(k+o3)*nMO+l+o4);
		  }
		}
	      }
	      for(int l=0;l<nVirt[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nc[q],(k+nc[r])*nMOs[s]+l+nOcc[s]+nc[s])=D2ab((i+o1)*nMO+j+o2,(k+o3)*nMO+l+v4+nels);
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nc[r])*nMOs[s]+l+nOcc[s]+nc[s])=D2ab((i+o1)*nMO+j+v2+nels,(k+o3)*nMO+l+v4+nels);
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nc[q],(k+nc[r])*nMOs[s]+l+nOcc[s]+nc[s])=D2ab((i+v1+nels)*nMO+j+o2,(k+o3)*nMO+l+v4+nels);
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nc[r])*nMOs[s]+l+nOcc[s]+nc[s])=D2ab((i+v1+nels)*nMO+j+v2+nels,(k+o3)*nMO+l+v4+nels);
		  }
		}
	      }
	    }
	    for(int k=0;k<nVirt[r];k++)
	    {
	      for(int l=0;l<nOcc[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nc[s])=D2ab((i+o1)*nMO+j+o2,(k+v3+nels)*nMO+l+o4);
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nc[s])=D2ab((i+o1)*nMO+j+v2+nels,(k+v3+nels)*nMO+l+o4);
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nc[s])=D2ab((i+v1+nels)*nMO+j+o2,(k+v3+nels)*nMO+l+o4);
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nc[s])=D2ab((i+v1+nels)*nMO+j+v2+nels,(k+v3+nels)*nMO+l+o4);
		  }
		}
	      }
	      for(int l=0;l<nVirt[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nOcc[s]+nc[s])=D2ab((i+o1)*nMO+j+o2,(k+v3+nels)*nMO+l+v4+nels);
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nOcc[s]+nc[s])=D2ab((i+o1)*nMO+j+v2+nels,(k+v3+nels)*nMO+l+v4+nels);
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nOcc[s]+nc[s])=D2ab((i+v1+nels)*nMO+j+o2,(k+v3+nels)*nMO+l+v4+nels);
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2abS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nOcc[s]+nc[s])=D2ab((i+v1+nels)*nMO+j+v2+nels,(k+v3+nels)*nMO+l+v4+nels);
		  }
		}
	      }
	    }
	  }

	  o4 += nOcc[s];
	  v4 += nVirt[s];
	}

	o3 += nOcc[r];	
	v3 += nVirt[r];	
      }
      o2 += nOcc[q];
      v2 += nVirt[q];
    }
    o1 += nOcc[p];
    v1 += nVirt[p];
  }


  o1=0; v1=0;
  for(int p=0;p<bas.size();p++)
  {
    o2=0; v2=0;
    for(int q=0;q<bas.size();q++)
    {
      for(int k=0;k<nOcc[q];k++)
      {
	for(int l=k+1;l<nOcc[q];l++)
	{
	  for(int i=0;i<nOcc[p];i++)
	  {
	    for(int j=i+1;j<nOcc[p];j++)
	    {
	      D2aaS[p*bas.size()+p][q*bas.size()+q](indexD(i+nc[p],j+nc[p],nMOs[p]),indexD(k+nc[q],l+nc[q],nMOs[q]))=D2aa(indexD(i+o1,j+o1,nMO),indexD(k+o2,l+o2,nMO));
	    }
	    for(int j=0;j<nVirt[p];j++)
	    {
	      D2aaS[p*bas.size()+p][q*bas.size()+q](indexD(i+nc[p],j+nOcc[p]+nc[p],nMOs[p]),indexD(k+nc[q],l+nc[q],nMOs[q]))=D2aa(indexD(i+o1,j+v1+nels,nMO),indexD(k+o2,l+o2,nMO));
	    }
	  }
	  for(int i=0;i<nVirt[p];i++)
	  {
	    for(int j=i+1;j<nVirt[p];j++)
	    {
	      D2aaS[p*bas.size()+p][q*bas.size()+q](indexD(i+nOcc[p]+nc[p],j+nOcc[p]+nc[p],nMOs[p]),indexD(k+nc[q],l+nc[q],nMOs[q]))=D2aa(indexD(i+v1+nels,j+v1+nels,nMO),indexD(k+o2,l+o2,nMO));
	    }
	  }
	}
	for(int l=0;l<nVirt[q];l++)
	{
	  for(int i=0;i<nOcc[p];i++)
	  {
	    for(int j=i+1;j<nOcc[p];j++)
	    {
	      D2aaS[p*bas.size()+p][q*bas.size()+q](indexD(i+nc[p],j+nc[p],nMOs[p]),indexD(k+nc[q],l+nOcc[q]+nc[q],nMOs[q]))=D2aa(indexD(i+o1,j+o1,nMO),indexD(k+o2,l+v2+nels,nMO));
	    }
	    for(int j=0;j<nVirt[p];j++)
	    {
	      D2aaS[p*bas.size()+p][q*bas.size()+q](indexD(i+nc[p],j+nOcc[p]+nc[p],nMOs[p]),indexD(k+nc[q],l+nOcc[q]+nc[q],nMOs[q]))=D2aa(indexD(i+o1,j+v1+nels,nMO),indexD(k+o2,l+v2+nels,nMO));
	    }
	  }
	  for(int i=0;i<nVirt[p];i++)
	  {
	    for(int j=i+1;j<nVirt[p];j++)
	    {
	      D2aaS[p*bas.size()+p][q*bas.size()+q](indexD(i+nOcc[p]+nc[p],j+nOcc[p]+nc[p],nMOs[p]),indexD(k+nc[q],l+nOcc[q]+nc[q],nMOs[q]))=D2aa(indexD(i+v1+nels,j+v1+nels,nMO),indexD(k+o2,l+v2+nels,nMO));
	    }
	  }
	}
      }
      for(int k=0;k<nVirt[q];k++)
      {
	for(int l=k+1;l<nVirt[q];l++)
	{
	  for(int i=0;i<nOcc[p];i++)
	  {
	    for(int j=i+1;j<nOcc[p];j++)
	    {
	      D2aaS[p*bas.size()+p][q*bas.size()+q](indexD(i+nc[p],j+nc[p],nMOs[p]),indexD(k+nOcc[q]+nc[q],l+nOcc[q]+nc[q],nMOs[q]))=D2aa(indexD(i+o1,j+o1,nMO),indexD(k+v2+nels,l+v2+nels,nMO));
	    }
	    for(int j=0;j<nVirt[p];j++)
	    {
	      D2aaS[p*bas.size()+p][q*bas.size()+q](indexD(i+nc[p],j+nOcc[p]+nc[p],nMOs[p]),indexD(k+nOcc[q]+nc[q],l+nOcc[q]+nc[q],nMOs[q]))=D2aa(indexD(i+o1,j+v1+nels,nMO),indexD(k+v2+nels,l+v2+nels,nMO));
	    }
	  }
	  for(int i=0;i<nVirt[p];i++)
	  {
	    for(int j=i+1;j<nVirt[p];j++)
	    {
	      D2aaS[p*bas.size()+p][q*bas.size()+q](indexD(i+nOcc[p]+nc[p],j+nOcc[p]+nc[p],nMOs[p]),indexD(k+nOcc[q]+nc[q],l+nOcc[q]+nc[q],nMOs[q]))=D2aa(indexD(i+v1+nels,j+v1+nels,nMO),indexD(k+v2+nels,l+v2+nels,nMO));
	    }
	  }
	}
      }

      o2 += nOcc[q];
      v2 += nVirt[q];
    }

    o2=o1+nOcc[p]; v2=v1+nVirt[p];
    for(int q=p+1;q<bas.size();q++)
    {
      o3=0; v3=0;
      for(int r=0;r<bas.size();r++)
      {
	o4=o3+nOcc[r]; v4=v3+nVirt[r];
	for(int s=r+1;s<bas.size();s++)
	{
	  if(D2abS[p*bas.size()+q][r*bas.size()+s].n_rows>0 && D2abS[p*bas.size()+q][r*bas.size()+s].n_cols>0)
	  {
	    for(int k=0;k<nOcc[r];k++)
	    {
	      for(int l=0;l<nOcc[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nc[q],(k+nc[r])*nMOs[s]+l+nc[s]) = D2aa(indexD(i+o1,j+o2,nMO),indexD(k+o3,l+o4,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nc[r])*nMOs[s]+l+nc[s]) = D2aa(indexD(i+o1,j+v2+nels,nMO),indexD(k+o3,l+o4,nMO));
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nc[q],(k+nc[r])*nMOs[s]+l+nc[s]) = -D2aa(indexD(j+o2,i+v1+nels,nMO),indexD(k+o3,l+o4,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nc[r])*nMOs[s]+l+nc[s]) = D2aa(indexD(i+v1+nels,j+v2+nels,nMO),indexD(k+o3,l+o4,nMO));
		  }
		}
	      }
	      for(int l=0;l<nVirt[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nc[q],(k+nc[r])*nMOs[s]+l+nOcc[s]+nc[s]) = D2aa(indexD(i+o1,j+o2,nMO),indexD(k+o3,l+v4+nels,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nc[r])*nMOs[s]+l+nOcc[s]+nc[s]) = D2aa(indexD(i+o1,j+v2+nels,nMO),indexD(k+o3,l+v4+nels,nMO));
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nc[q],(k+nc[r])*nMOs[s]+l+nOcc[s]+nc[s]) = -D2aa(indexD(j+o2,i+v1+nels,nMO),indexD(k+o3,l+v4+nels,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nc[r])*nMOs[s]+l+nOcc[s]+nc[s]) = D2aa(indexD(i+v1+nels,j+v2+nels,nMO),indexD(k+o3,l+v4+nels,nMO));
		  }
		}
	      }
	    }
	    for(int k=0;k<nVirt[r];k++)
	    {
	      for(int l=0;l<nOcc[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nc[s]) = -D2aa(indexD(i+o1,j+o2,nMO),indexD(l+o4,k+v3+nels,nMO));
		    //D2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j,(k+nOcc[r])*nMOs[s]+l) = D2aa(indexD(i+o1,j+o2,nMO),indexD(k+o3,l+o4,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nc[s]) = -D2aa(indexD(i+o1,j+v2+nels,nMO),indexD(l+o4,k+v3+nels,nMO));
		    //D2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j+nOcc[q],(k+nOcc[r])*nMOs[s]+l) = D2aa(indexD(i+o1,j+v2+nels,nMO),indexD(k+o3,l+o4,nMO));
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nc[s]) = D2aa(indexD(j+o2,i+v1+nels,nMO),indexD(l+o4,k+v3+nels,nMO));
		    //D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p])*nMOs[q]+j,(k+nOcc[r])*nMOs[s]+l) = D2aa(indexD(j+o2,i+v1+nels,nMO),indexD(k+o3,l+o4,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nc[s]) = -D2aa(indexD(i+v1+nels,j+v2+nels,nMO),indexD(l+o4,k+v3+nels,nMO));
		    //D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p])*nMOs[q]+j+nOcc[q],(k+nOcc[r])*nMOs[s]+l) = D2aa(indexD(i+v1+nels,j+v2+nels,nMO),indexD(l+o4,k+v3+nels,nMO));
		  }
		}
	      }
	      for(int l=0;l<nVirt[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nOcc[s]+nc[s]) = D2aa(indexD(i+o1,j+o2,nMO),indexD(k+v3+nels,l+v4+nels,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nOcc[s]+nc[s]) = D2aa(indexD(i+o1,j+v2+nels,nMO),indexD(k+v3+nels,l+v4+nels,nMO));
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nOcc[s]+nc[s]) = -D2aa(indexD(j+o2,i+v1+nels,nMO),indexD(k+v3+nels,l+v4+nels,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p]+nc[p])*nMOs[q]+j+nOcc[q]+nc[q],(k+nOcc[r]+nc[r])*nMOs[s]+l+nOcc[s]+nc[s]) = D2aa(indexD(i+v1+nels,j+v2+nels,nMO),indexD(k+v3+nels,l+v4+nels,nMO));
		  }
		}
	      }
	    }
	    /*for(int k=0;k<nOcc[r];k++)
	    {
	      for(int l=0;l<nOcc[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j,k*nMOs[s]+l) = D2aa(indexD(i+o1,j+o2,nMO),indexD(k+o3,l+o4,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j+nOcc[q],k*nMOs[s]+l) = D2aa(indexD(i+o1,j+v2+nels,nMO),indexD(k+o3,l+o4,nMO));
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p])*nMOs[q]+j,k*nMOs[s]+l) = D2aa(indexD(i+v1+nels,j+o2,nMO),indexD(k+o3,l+o4,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p])*nMOs[q]+j+nOcc[q],k*nMOs[s]+l) = D2aa(indexD(i+v1+nels,j+v2+nels,nMO),indexD(k+o3,l+o4,nMO));
		  }
		}
	      }
	      for(int l=0;l<nVirt[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j,k*nMOs[s]+l+nOcc[s]) = D2aa(indexD(i+o1,j+o2,nMO),indexD(k+o3,l+v4+nels,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j+nOcc[q],k*nMOs[s]+l+nOcc[s]) = D2aa(indexD(i+o1,j+v2+nels,nMO),indexD(k+o3,l+v4+nels,nMO));
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p])*nMOs[q]+j,k*nMOs[s]+l+nOcc[s]) = D2aa(indexD(i+v1+nels,j+o2,nMO),indexD(k+o3,l+v4+nels,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p])*nMOs[q]+j+nOcc[q],k*nMOs[s]+l+nOcc[s]) = D2aa(indexD(i+v1+nels,j+v2+nels,nMO),indexD(k+o3,l+v4+nels,nMO));
		  }
		}
	      }
	    }
	    for(int k=0;k<nVirt[r];k++)
	    {
	      for(int l=0;l<nOcc[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j,(k+nOcc[r])*nMOs[s]+l) = D2aa(indexD(i+o1,j+o2,nMO),indexD(k+v3+nels,l+o4,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j+nOcc[q],(k+nOcc[r])*nMOs[s]+l) = D2aa(indexD(i+o1,j+v2+nels,nMO),indexD(k+v3+nels,l+o4,nMO));
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p])*nMOs[q]+j,(k+nOcc[r])*nMOs[s]+l) = D2aa(indexD(i+v1+nels,j+o2,nMO),indexD(k+v3+nels,l+o4,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p])*nMOs[q]+j+nOcc[q],(k+nOcc[r])*nMOs[s]+l) = D2aa(indexD(i+v1+nels,j+v2+nels,nMO),indexD(k+v3+nels,l+o4,nMO));
		  }
		}
	      }
	      for(int l=0;l<nVirt[s];l++)
	      {
		for(int i=0;i<nOcc[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j,(k+nOcc[r])*nMOs[s]+l+nOcc[s]) = D2aa(indexD(i+o1,j+o2,nMO),indexD(k+v3+nels,l+v4+nels,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j+nOcc[q],(k+nOcc[r])*nMOs[s]+l+nOcc[s]) = D2aa(indexD(i+o1,j+v2+nels,nMO),indexD(k+v3+nels,l+v4+nels,nMO));
		  }
		}
		for(int i=0;i<nVirt[p];i++)
		{
		  for(int j=0;j<nOcc[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p])*nMOs[q]+j,(k+nOcc[r])*nMOs[s]+l+nOcc[s]) = D2aa(indexD(i+v1+nels,j+o2,nMO),indexD(k+v3+nels,l+v4+nels,nMO));
		  }
		  for(int j=0;j<nVirt[q];j++)
		  {
		    D2aaS[p*bas.size()+q][r*bas.size()+s]((i+nOcc[p])*nMOs[q]+j+nOcc[q],(k+nOcc[r])*nMOs[s]+l+nOcc[s]) = D2aa(indexD(i+v1+nels,j+v2+nels,nMO),indexD(k+v3+nels,l+v4+nels,nMO));
		  }
		}
	      }
	    }*/
	  }
	
	  o4 += nOcc[s];
	  v4 += nVirt[s];
	}

	o3 += nOcc[r];
	v3 += nVirt[r];
      }

      o2 += nOcc[q];
      v2 += nVirt[q];
    }

    o1 += nOcc[p];
    v1 += nVirt[p];
  }

  return;
}

void makeV2aa(const std::vector<std::vector<arma::mat> > &V2ab,std::vector<std::vector<arma::mat> > &V2aaS,const std::vector<std::vector<std::vector<int> > > &bas,std::vector<int> nMOs)
{
  int o1,o2,o3,o4;
  int v1,v2,v3,v4;
  V2aaS.resize(bas.size()*bas.size());
  for(int p=0;p<V2aaS.size();p++)
  {
    V2aaS[p].resize(V2aaS.size());
  }

  for(int p=0;p<bas.size();p++)
  {
    for(int q=0;q<bas.size();q++)
    {
      V2aaS[p*bas.size()+p][q*bas.size()+q].set_size(nMOs[p]*(nMOs[p]-1)/2,nMOs[q]*(nMOs[q]-1)/2);
      V2aaS[p*bas.size()+p][q*bas.size()+q].zeros();
    }
    for(int q=p+1;q<bas.size();q++)
    {
      for(int r=0;r<bas.size();r++)
      {
	for(int s=r+1;s<bas.size();s++)
	{
	  V2aaS[p*bas.size()+q][r*bas.size()+s].set_size(nMOs[p]*nMOs[q],nMOs[r]*nMOs[s]);
	  V2aaS[p*bas.size()+q][r*bas.size()+s].zeros();
	}
      }
    }
  }

  double x;
  for(int p=0;p<bas.size();p++)
  {
    for(int q=0;q<bas.size();q++)
    {
      for(int k=0;k<nMOs[q];k++)
      {
	for(int l=k+1;l<nMOs[q];l++)
	{
	  for(int i=0;i<nMOs[p];i++)
	  {
	    for(int j=i+1;j<nMOs[p];j++)
	    {
	      x=V2ab[p*bas.size()+p][q*bas.size()+q](i*nMOs[p]+j,k*nMOs[q]+l);
	      x-=V2ab[p*bas.size()+p][q*bas.size()+q](j*nMOs[p]+i,k*nMOs[q]+l);
	      V2aaS[p*bas.size()+p][q*bas.size()+q](indexD(i,j,nMOs[p]),indexD(k,l,nMOs[q])) = 2.0*x;
	    }
	  }
	}
      }
    }

    for(int q=p+1;q<bas.size();q++)
    {
      for(int r=0;r<bas.size();r++)
      {
	for(int s=r+1;s<bas.size();s++)
	{
	  if(V2ab[p*bas.size()+q][r*bas.size()+s].n_rows>0 && V2ab[p*bas.size()+q][r*bas.size()+s].n_cols>0)
	  {
	    for(int k=0;k<nMOs[r];k++)
	    {
	      for(int l=0;l<nMOs[s];l++)
	      {
		for(int i=0;i<nMOs[p];i++)
		{
		  for(int j=0;j<nMOs[q];j++)
		  {
		    x=V2ab[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j,k*nMOs[s]+l);
		    x-=V2ab[q*bas.size()+p][r*bas.size()+s](j*nMOs[p]+i,k*nMOs[s]+l);
		    V2aaS[p*bas.size()+q][r*bas.size()+s](i*nMOs[q]+j,k*nMOs[s]+l) = 2.0*x;
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

void gradEvalZ(const real_1d_array &R,double &E,real_1d_array &grad,void *ptr)
{
  std::chrono::high_resolution_clock::time_point t1,t2;
  std::chrono::duration<double> time_span;

  setTime(t1,t1,time_span);
  int nT=0;
  int nT2;
  std::string label;
  
  double x;
  bundleNuc *bPtr=static_cast<bundleNuc*>(ptr);
  for(int m=0;m<bPtr->modes.size();m++)
  {
    switch(bPtr->modes[m][1])
    {
      case 0:
	bPtr->atoms[(*bPtr).modes[m][0]].x = R[m];
	break;
      case 1:
	bPtr->atoms[(*bPtr).modes[m][0]].y = R[m];
	break;
      case 2:
	bPtr->atoms[(*bPtr).modes[m][0]].z = R[m];
	break;
    }
  }
  bPtr->numE++;

  std::cout << "\nUnique atoms (angstrom):" << std::endl;
  std::cout << "Z\tx\t\ty\t\tz\n";

  double bohr= 0.52917721092;

  for(int n=0;n<(*bPtr).atoms.size();n++)
  {
    printf("%d\t%.9f\t%.9f\t%.9f\n",bPtr->atoms[n].atomic_number,bPtr->atoms[n].x*bohr,bPtr->atoms[n].y*bohr,bPtr->atoms[n].z*bohr);
  }

  std::vector<libint2::Atom> allAtoms;
  genAtoms((*bPtr).basis,(*bPtr).pointGroup,(*bPtr).atoms,allAtoms);

  //arma::mat T1,T2aa,T2ab;
  std::vector<arma::mat> K1s;
  std::vector<std::vector<arma::mat> > V2s;
  double Ep;
  bool useOld=true;
  if(bPtr->oldE == 0.0)
  {
    useOld=false;
  }

  std::fstream punchFile;
  punchFile.open(bPtr->punchName,std::fstream::app);
  punchFile << "Iter:\t" << bPtr->numIt+1 << std::endl;
  punchFile.close();

  runPara(allAtoms,(*bPtr).basis,(*bPtr).nAOs,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).prodTable,Ep,bPtr->T1,bPtr->T2aa,bPtr->T2ab,K1s,V2s,bPtr->Cs,bPtr->nOcc,bPtr->nCore,true,true,bPtr->core,bPtr->numPe,useOld,bPtr->punchName,bPtr->fail);

  if(bPtr->fail)
  {
    runPara(allAtoms,(*bPtr).basis,(*bPtr).nAOs,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).prodTable,Ep,bPtr->T1,bPtr->T2aa,bPtr->T2ab,K1s,V2s,bPtr->Cs,bPtr->nOcc,bPtr->nCore,true,true,bPtr->core,bPtr->numPe,false,bPtr->punchName,bPtr->fail);
  }

  //runPara(allAtoms,(*bPtr).basis,(*bPtr).nAOs,(*bPtr).ao2mo,(*bPtr).bas,(*bPtr).prodTable,Ep,bPtr->T1,bPtr->T2aa,bPtr->T2ab,K1s,V2s,bPtr->Cs,bPtr->nOcc,bPtr->nCore,true,false,bPtr->core,bPtr->numPe);

  std::vector<int> nCore=bPtr->nCore;
  E=Ep;
  (*bPtr).E=E;

  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "Single point";
  nT++; t1=t2;

  arma::mat D1,D2aa,D2ab;
  makeD2(bPtr->T1,bPtr->T2aa,bPtr->T2ab,D1,D2aa,D2ab);

  arma::vec eigval;
  arma::mat eigvec;
  eig_sym(eigval,eigvec,D1);
  punchFile.open(bPtr->punchName,std::fstream::app);
  char cstr[100];
  std::string str;
  punchFile << "Occupation Numbers:\n";
  for(int i=0;i<eigval.size();i+=10)
  {
    for(int j=i;j<std::min(i+10,int(eigval.size()));j++)
    {
      sprintf(cstr,"%.4f\t",eigval(eigval.size()-j-1));
      str=std::string(cstr);
      punchFile << str;
    }
    punchFile << std::endl;
  }
  punchFile << std::endl;
  punchFile.close();


  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "make D2";
  nT++; t1=t2;

  std::vector<int> nMO(bPtr->nOcc.size());
  for(int p=0;p<bPtr->nOcc.size();p++)
  {
    nMO[p]=K1s[p].n_cols;
  }

  std::vector<arma::mat> D1s;
  std::vector<std::vector<arma::mat> > D2aaS,D2abS;
  rotateForward(bPtr->nOcc,D1,D2aa,D2ab,nMO,D1s,D2aaS,D2abS,(*bPtr).bas,nCore);

  std::vector<std::vector<arma::mat> > V2aaS;
  makeV2aa(V2s,V2aaS,(*bPtr).bas,nMO);

  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "make V2aa";
  nT++; t1=t2;

  std::vector<int> nVirt;
  for(int p=0;p<bPtr->nOcc.size();p++)
  {
    nVirt.push_back(nMO[p]-bPtr->nOcc[p]);
  }

  int nels=bPtr->T1.n_rows;
  int virt=bPtr->T1.n_cols;
  int nact=nels+virt;

  std::vector<int> nOcc=bPtr->nOcc;
  std::vector<std::vector<double> > eps(bPtr->nOcc.size());
  for(int p=0;p<bPtr->nOcc.size();p++)
  {
    for(int i=0;i<K1s[p].n_cols;i++)
    {
      x=K1s[p](i,i);
      for(int q=0;q<bPtr->nOcc.size();q++)
      {
	for(int j=0;j<bPtr->nOcc[q]+nCore[q];j++)
	{
	  x += 2.0*V2s[p*nOcc.size()+q][p*nOcc.size()+q](i*K1s[q].n_cols+j,i*K1s[q].n_cols+j);
	  x -= V2s[q*nOcc.size()+p][p*nOcc.size()+q](j*K1s[p].n_cols+i,i*K1s[q].n_cols+j);
	}
      }
      eps[p].push_back(x);
    }
  }
  arma::vec X;
  arma::vec Z;

  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "make eps";
  nT++; t1=t2;

  for(int p=0;p<nCore.size();p++)
  {
    nOcc[p] += nCore[p];
  }

  std::vector<std::vector<std::vector<int> > > nonDeg;
  makeZ2(D1s,D2aaS,D2abS,K1s,V2aaS,V2s,eps,nMO,nOcc,X,Z,nonDeg);

  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "make Z";
  nT++; t1=t2;

  rotateMO_AO(D1s,D2aaS,D2abS,bPtr->Cs,nMO);

  setTime(t1,t2,time_span);
  bPtr->ts[nT] += time_span;
  bPtr->tLabel[nT] = "rotate D2";
  nT++; t1=t2;

  double EnucP,EnucM;
  std::vector<arma::mat> Sp,CsP,K1p;
  std::vector<std::vector<arma::mat> > V2p;

  std::vector<std::vector<arma::mat> > V2aaP;
  double dx=0.000001;
  double Em;

  arma::vec B,dS;

  for(int m=0;m<(*bPtr).modes.size();m++)
  {
    setTime(t1,t1,time_span);

    genIntsDer(EnucP,bPtr->atoms,bPtr->basis,Sp,K1p,V2p,bPtr->ao2mo,bPtr->bas,bPtr->nAOs,bPtr->prodTable,bPtr->pointGroup,bPtr->modes[m],dx);

    setTime(t1,t2,time_span);
    label = "make deriv"; findLabel(label,bPtr->tLabel,nT);
    bPtr->tLabel[nT] = label;
    bPtr->ts[nT] += time_span;
    nT++; t1=t2;

    makeB(eps,bPtr->Cs,Sp,K1p,V2p,nMO,nOcc,V2s,B,dS,nonDeg);

    setTime(t1,t2,time_span);
    label = "make B"; findLabel(label,bPtr->tLabel,nT);
    bPtr->tLabel[nT] = label;
    bPtr->ts[nT] += time_span;
    nT++; t1=t2;

    x=0.0;
    for(int p=0;p<D1s.size();p++)
    {
      x += 2.0*trace(K1p[p]*D1s[p]);
    }
      for(int p=0;p<D1s.size();p++)
      {
	for(int q=0;q<D1s.size();q++)
	{
	  for(int r=0;r<D1s.size();r++)
	  {
	    for(int s=0;s<D1s.size();s++)
	    {
	      x += arma::accu(V2p[p*D1s.size()+q][r*D1s.size()+s]%D2abS[p*D1s.size()+q][r*D1s.size()+s]);
	    }
	  }
	}
      }
      makeV2aa(V2p,V2aaP,(*bPtr).bas,nMO);
      for(int p=0;p<D1s.size();p++)
      {
	for(int q=p;q<D1s.size();q++)
	{
	  for(int r=0;r<D1s.size();r++)
	  {
	    for(int s=r;s<D1s.size();s++)
	    {
	      x += arma::accu(V2aaP[p*D1s.size()+q][r*D1s.size()+s]%D2aaS[p*D1s.size()+q][r*D1s.size()+s]);
	    }
	  }
	}
      }

    setTime(t1,t2,time_span);
    label = "K2D2"; findLabel(label,bPtr->tLabel,nT);
    bPtr->tLabel[nT] = label;
    bPtr->ts[nT] += time_span;
    nT++; t1=t2;

    grad[m]=EnucP+x+2.0*arma::dot(Z,B)-2.0*arma::dot(dS,X);
    bPtr->g[m]=grad[m];

//std::cout << m << '\t' << grad[m] << std::endl;

    setTime(t1,t2,time_span);
    label = "final evaluation"; findLabel(label,bPtr->tLabel,nT);
    bPtr->tLabel[nT] = label;
    bPtr->ts[nT] += time_span;
    nT++; t1=t2;

  }

  return;
}

void rotateMO_AO(std::vector<arma::mat> &D1,std::vector<std::vector<arma::mat> > &D2aa,std::vector<std::vector<arma::mat> > &D2ab,const std::vector<arma::mat> &Cs,const std::vector<int> &nMO)
{
  for(int p=0;p<D1.size();p++)
  {
    D1[p] = Cs[p]*D1[p]*Cs[p].t();
  }

  for(int p=0;p<D1.size();p++)
  {
    for(int q=0;q<D1.size();q++)
    {
      for(int r=0;r<D1.size();r++)
      {
	for(int s=0;s<D1.size();s++)
	{
	  if(D2ab[p*D1.size()+q][r*D1.size()+s].n_rows>0 && D2ab[p*D1.size()+q][r*D1.size()+s].n_cols>0)
	  {
	    D2ab[p*D1.size()+q][r*D1.size()+s] = arma::kron(Cs[p],Cs[q])*D2ab[p*D1.size()+q][r*D1.size()+s]*arma::kron(Cs[r].t(),Cs[s].t());
	  }
	}
      }
    }
  }

  for(int p=0;p<D1.size();p++)
  {
    for(int q=p+1;q<D1.size();q++)
    {
      for(int r=0;r<D1.size();r++)
      {
	for(int s=r+1;s<D1.size();s++)
	{
	  if(D2ab[p*D1.size()+q][r*D1.size()+s].n_rows>0 && D2ab[p*D1.size()+q][r*D1.size()+s].n_cols>0)
	  {
	    D2aa[p*D1.size()+q][r*D1.size()+s] = arma::kron(Cs[p],Cs[q])*D2aa[p*D1.size()+q][r*D1.size()+s]*arma::kron(Cs[r].t(),Cs[s].t());
	  }
	}
      }
    }
  }

  arma::mat C1,C2;
  for(int p=0;p<D1.size();p++)
  {
    C1.set_size(Cs[p].n_rows*(Cs[p].n_rows-1)/2,Cs[p].n_cols*(Cs[p].n_cols-1)/2);
    for(int i=0;i<Cs[p].n_cols;i++)
    {
      for(int j=i+1;j<Cs[p].n_cols;j++)
      {
	for(int a=0;a<Cs[p].n_rows;a++)
	{
	  for(int b=a+1;b<Cs[p].n_rows;b++)
	  {
	    C1(indexD(a,b,Cs[p].n_rows),indexD(i,j,Cs[p].n_cols)) = Cs[p](a,i)*Cs[p](b,j)-Cs[p](b,i)*Cs[p](a,j);
	  }
	}
      }
    }
    for(int q=0;q<D1.size();q++)
    {
      C2.set_size(Cs[q].n_rows*(Cs[q].n_rows-1)/2,Cs[q].n_cols*(Cs[q].n_cols-1)/2);
      for(int i=0;i<Cs[q].n_cols;i++)
      {
	for(int j=i+1;j<Cs[q].n_cols;j++)
	{
	  for(int a=0;a<Cs[q].n_rows;a++)
	  {
	    for(int b=a+1;b<Cs[q].n_rows;b++)
	    {
	      C2(indexD(a,b,Cs[q].n_rows),indexD(i,j,Cs[q].n_cols)) = Cs[q](a,i)*Cs[q](b,j)-Cs[q](b,i)*Cs[q](a,j);
	    }
	  }
	}
      }
      D2aa[p*D1.size()+p][q*D1.size()+q] = C1*D2aa[p*D1.size()+p][q*D1.size()+q]*C2.t();
    }
  }

  return;
}

void makeB(const std::vector<std::vector<double> > &eps,const std::vector<arma::mat> &Cs,const std::vector<arma::mat> &S,const std::vector<arma::mat> &K1,const std::vector<std::vector<arma::mat> > &V2,const std::vector<int> &nMO,const std::vector<int> &nOcc,const std::vector<std::vector<arma::mat> > &V2s,arma::vec &B,arma::vec &dS,const std::vector<std::vector<std::vector<int> > > &nonDeg)
{
  std::vector<arma::mat> dSr(S.size());
  for(int p=0;p<S.size();p++)
  {
    dSr[p] = Cs[p].t()*S[p]*Cs[p];
  }

  int nTot=0;
  for(int p=0;p<S.size();p++)
  {
    nTot += nMO[p]*(nMO[p]+1)/2;
  }
  
  dS.set_size(nTot);
  int num1=0;
  for(int p=0;p<S.size();p++)
  {
    for(int i=0;i<nMO[p];i++)
    {
      dS(num1) = dSr[p](i,i);
      num1++;
      for(int j=0;j<i;j++)
      {
	dS(num1) = dSr[p](i,j);
	num1++;
      }
    }
  }
  
  double x,y;

  std::vector<arma::mat> D1;
  for(int p=0;p<K1.size();p++)
  {
    D1.push_back(arma::mat(K1[p].n_rows,K1[p].n_cols,arma::fill::zeros));
    for(int i=0;i<nOcc[p];i++)
    {
      D1[p] += arma::kron(Cs[p].col(i).t(),Cs[p].col(i));
    }
  }
 
  std::vector<arma::mat> dF;
  for(int p=0;p<D1.size();p++)
  {
    dF.push_back(K1[p]);
    for(int i=0;i<K1[p].n_rows;i++)
    {
      for(int j=0;j<K1[p].n_rows;j++)
      {
	x=0.0;
	for(int q=0;q<D1.size();q++)
	{
	  for(int k=0;k<K1[q].n_rows;k++)
	  {
	    for(int l=0;l<K1[q].n_rows;l++)
	    {
	      y = 2.0*V2[p*D1.size()+q][p*D1.size()+q](i*K1[q].n_rows+k,j*K1[q].n_rows+l)-V2[p*D1.size()+q][q*D1.size()+p](i*K1[q].n_rows+k,l*K1[p].n_rows+j);
	      x += y*D1[q](k,l);
	    }
	  }
	}
	dF[p](i,j) += x;
      }
    }
    dF[p] = Cs[p].t()*dF[p]*Cs[p];
  }

  nTot=0;
  std::vector<int> nVirt(nOcc.size());
  for(int p=0;p<S.size();p++)
  {
    nVirt[p] = nMO[p]-nOcc[p];
    nTot += nonDeg[p].size();
    nTot += nVirt[p]*nOcc[p];
  }

  B.set_size(nTot);

  num1=0;
  int i,j;
  for(int p=0;p<S.size();p++)
  {
    for(int t=0;t<nonDeg[p].size();t++)
    {
      i=nonDeg[p][t][0];
      j=nonDeg[p][t][1];
      B(num1+t) = dF[p](i,j);
      B(num1+t) -= dSr[p](i,j)*eps[p][j];
      x=0.0;
      for(int q=0;q<S.size();q++)
      {
	for(int k=0;k<nOcc[q];k++)
	{
	  for(int l=0;l<nOcc[q];l++)
	  {
	    y = 2.0*V2s[p*S.size()+q][p*S.size()+q](i*nMO[q]+k,j*nMO[q]+l)-V2s[p*S.size()+p][q*S.size()+q](i*nMO[p]+j,k*nMO[q]+l);
	    x += y*dSr[q](k,l);
	  }
	}	
      }
      B(num1+t) -= x;
    }
    num1 += nonDeg[p].size();
    for(int a=0;a<nVirt[p];a++)
    {
      for(int k=0;k<nOcc[p];k++)
      {
	B(num1+a*nOcc[p]+k) = dF[p](a+nOcc[p],k);
	B(num1+a*nOcc[p]+k) -= dSr[p](a+nOcc[p],k)*eps[p][k];
	x=0.0;
	for(int q=0;q<S.size();q++)
	{
	  for(int m=0;m<nOcc[q];m++)
	  {
	    for(int l=0;l<nOcc[q];l++)
	    {
	      y = 2.0*V2s[p*S.size()+q][p*S.size()+q]((a+nOcc[p])*nMO[q]+m,k*nMO[q]+l)-V2s[p*S.size()+p][q*S.size()+q]((a+nOcc[p])*nMO[p]+k,m*nMO[q]+l);
	      x += y*dSr[q](m,l);
	    }
	  }	
	}
	B(num1+a*nOcc[p]+k) -= x;
      }
    }
    num1 += nOcc[p]*nVirt[p];
  }

  return;
}

void makeZ2(const std::vector<arma::mat> &D1,const std::vector<std::vector<arma::mat> > &D2aa,const std::vector<std::vector<arma::mat> > &D2ab,const std::vector<arma::mat> &K1,const std::vector<std::vector<arma::mat> > &V2aa,const std::vector<std::vector<arma::mat> > &V2ab,const std::vector<std::vector<double> > &eps,const std::vector<int> &nMO,const std::vector<int> &nOcc,arma::vec &dX,arma::vec &Z,std::vector<std::vector<std::vector<int> > > &nonDeg)
{
  std::vector<arma::mat> X(D1.size());
  for(int p=0;p<D1.size();p++)
  {
    X[p].set_size(nMO[p],nMO[p]);
    X[p].zeros();
  }

  double x;
  for(int p=0;p<D1.size();p++)
  {
    for(int j=0;j<nMO[p];j++)
    {
      for(int i=0;i<nMO[p];i++)
      {
	x=0.0;
	for(int m=0;m<nMO[p];m++)
	{
	  x += D1[p](m,i)*K1[p](m,j);
	}
	X[p](i,j) += 2.0*x;
      }
    }
  }

  double y;
  arma::mat W;
  for(int p=0;p<D1.size();p++)
  {
    for(int q=0;q<D1.size();q++)
    {
      for(int r=0;r<D1.size();r++)
      {
	for(int s=0;s<D1.size();s++)
	{
	  if(V2ab[p*D1.size()+q][r*D1.size()+s].n_rows>0 && V2ab[p*D1.size()+q][r*D1.size()+s].n_cols>0)
	  {
	    W = D2ab[p*D1.size()+q][r*D1.size()+s]*V2ab[r*D1.size()+s][p*D1.size()+q];
	    for(int j=0;j<nMO[p];j++)
	    {
	      for(int i=0;i<nMO[p];i++)
	      {
		x=0.0;
		for(int m=0;m<nMO[q];m++)
		{
		  x += W(i*nMO[q]+m,j*nMO[q]+m);
		}
		X[p](i,j) += 2.0*x;
	      }
	    }
	  }
	}
      }
    }
  }
/*
  for(int p=0;p<D1.size();p++)
  {
    for(int j=0;j<nMO[p];j++)
    {
      for(int i=0;i<nMO[p];i++)
      {
 	x=0.0;
	for(int q=0;q<D1.size();q++)
	{
	  for(int r=0;r<D1.size();r++)
	  {
	    for(int s=0;s<D1.size();s++)
	    {
	      if(V2ab[r*D1.size()+s][p*D1.size()+q].n_rows>0 && V2ab[r*D1.size()+s][p*D1.size()+q].n_cols>0)
	      {
		for(int m=0;m<nMO[q];m++)
		{
		  for(int k=0;k<nMO[r];k++)
		  {
		    for(int l=0;l<nMO[s];l++)
		    {
		      y = D2ab[r*D1.size()+s][p*D1.size()+q](k*nMO[s]+l,i*nMO[q]+m);
		      x += y*V2ab[r*D1.size()+s][p*D1.size()+q](k*nMO[s]+l,j*nMO[q]+m);;
		    }
		  }
		}
	      }
	    }
	  }
	}
	X[p](i,j) += 2.0*x;
      }
    }
  }
*/
  for(int p=0;p<D1.size();p++)
  {
    for(int q=0;q<D1.size();q++)
    {
      W = D2aa[p*D1.size()+p][q*D1.size()+q]*V2aa[q*D1.size()+q][p*D1.size()+p];
      for(int j=0;j<nMO[p];j++)
      {
	for(int i=0;i<nMO[p];i++)
	{
	  x=0.0;
	  for(int m=0;m<std::min(i,j);m++)
	  {
	    x += W(indexD(m,i,nMO[p]),indexD(m,j,nMO[p]));
	  }
	  for(int m=i+1;m<j;m++)
	  {
	    x -= W(indexD(i,m,nMO[p]),indexD(m,j,nMO[p]));
	  }
	  for(int m=j+1;m<i;m++)
	  {
	    x -= W(indexD(m,i,nMO[p]),indexD(j,m,nMO[p]));
	  }
	  for(int m=std::max(i,j)+1;m<nMO[p];m++)
	  {
	    x += W(indexD(i,m,nMO[p]),indexD(j,m,nMO[p]));
	  }
	  X[p](i,j) += x;
	}
      }
    }
  }

  for(int p=0;p<D1.size();p++)
  {
    for(int q=p+1;q<D1.size();q++)
    {
      for(int r=0;r<D1.size();r++)
      {
	for(int s=r+1;s<D1.size();s++)
	{
	  if(V2ab[p*D1.size()+q][r*D1.size()+s].n_rows>0 && V2ab[p*D1.size()+q][r*D1.size()+s].n_cols>0)
	  {
	    W = D2aa[p*D1.size()+q][r*D1.size()+s]*V2aa[r*D1.size()+s][p*D1.size()+q];
	    for(int j=0;j<nMO[p];j++)
	    {
	      for(int i=0;i<nMO[p];i++)
	      {
		x=0.0;
		for(int m=0;m<nMO[q];m++)
		{
		  x += W(i*nMO[q]+m,j*nMO[q]+m);
		}
		X[p](i,j) += x;
	      }
	    }
	    for(int j=0;j<nMO[q];j++)
	    {
	      for(int i=0;i<nMO[q];i++)
	      {
		x=0.0;
		for(int m=0;m<nMO[p];m++)
		{
		  x += W(m*nMO[q]+i,m*nMO[q]+j);
		}
		X[q](i,j) += x;
	      }
	    }
	  }
	}
      }
    }
  }

  std::vector<int> nVirt(nOcc.size());
  for(int p=0;p<D1.size();p++)
  {
    nVirt[p] = nMO[p]-nOcc[p];
  }

  double d;
  nonDeg.resize(X.size());
  std::vector<int> v(2);

  for(int p=0;p<X.size();p++)
  {
    for(int i=0;i<nOcc[p];i++)
    {
      for(int j=0;j<i;j++)
      {
	d = X[p](j,i)-X[p](i,j);
	if(fabs(d) > 0.00000001)
	{
	  v[0]=i;
	  v[1]=j;
	  nonDeg[p].push_back(v);
	}
      }
    }
    for(int a=nOcc[p];a<nMO[p];a++)
    {
      for(int b=nOcc[p];b<a;b++)
      {
	d = X[p](b,a)-X[p](a,b);
	if(fabs(d) > 0.00000001)
	{
	  v[0]=a;
	  v[1]=b;
	  nonDeg[p].push_back(v);
	}
      }
    }
  }

  int nIA=0;
  int nNonDeg=0;
  for(int p=0;p<D1.size();p++)
  {
    nIA += nOcc[p]*nVirt[p];
    nNonDeg += nonDeg[p].size();
  }

  arma::mat Aind(nIA,nNonDeg);
  arma::mat Aov(nIA,nIA);

  int num1=0;
  int num2;
  for(int q=0;q<D1.size();q++)
  {
    num2=0;
    for(int p=0;p<D1.size();p++)
    {
      for(int a=0;a<nVirt[q];a++)
      {
	for(int i=0;i<nOcc[q];i++)
	{
	  for(int b=0;b<nVirt[p];b++)
	  {
	    for(int j=0;j<nOcc[p];j++)
	    {
	      Aov(num2+b*nOcc[p]+j,num1+a*nOcc[q]+i) = -4.0*V2ab[p*D1.size()+q][p*D1.size()+q]((b+nOcc[p])*nMO[q]+i,j*nMO[q]+a+nOcc[q]);
	      Aov(num2+b*nOcc[p]+j,num1+a*nOcc[q]+i) += V2ab[p*D1.size()+p][q*D1.size()+q]((b+nOcc[p])*nMO[p]+j,i*nMO[q]+a+nOcc[q]);
	      Aov(num2+b*nOcc[p]+j,num1+a*nOcc[q]+i) += V2ab[p*D1.size()+p][q*D1.size()+q](j*nMO[p]+b+nOcc[p],i*nMO[q]+a+nOcc[q]);
	    }
	  }
	}
      }
      num2 += nOcc[p]*nVirt[p];
    }
    for(int a=0;a<nVirt[q];a++)
    {
      for(int i=0;i<nOcc[q];i++)
      {
	Aov(num1+a*nOcc[q]+i,num1+a*nOcc[q]+i) += eps[q][i]-eps[q][a+nOcc[q]];
      }
    }

    num1 += nOcc[q]*nVirt[q];    
  }

  num1=0;
  int i,j;
  for(int q=0;q<D1.size();q++)
  {
    num2=0;
    for(int p=0;p<D1.size();p++)
    {
      for(int t=0;t<nonDeg[q].size();t++)
      {
	int a=nonDeg[q][t][0];
	i=nonDeg[q][t][1];
	for(int b=0;b<nVirt[p];b++)
	{
	  for(int j=0;j<nOcc[p];j++)
	  {
	    Aind(num2+b*nOcc[p]+j,num1+t) = -4.0*V2ab[p*D1.size()+q][p*D1.size()+q]((b+nOcc[p])*nMO[q]+i,j*nMO[q]+a);
	    Aind(num2+b*nOcc[p]+j,num1+t) += V2ab[p*D1.size()+p][q*D1.size()+q]((b+nOcc[p])*nMO[p]+j,i*nMO[q]+a);
	    Aind(num2+b*nOcc[p]+j,num1+t) += V2ab[p*D1.size()+p][q*D1.size()+q](j*nMO[p]+b+nOcc[p],i*nMO[q]+a);
	  }
	}
      }
      num2 += nOcc[p]*nVirt[p];
    }

    num1 += nonDeg[q].size();    
  }

  arma::vec Zind(nNonDeg);

  num1=0;
  for(int p=0;p<D1.size();p++)
  {
    for(int t=0;t<nonDeg[p].size();t++)
    {
      i=nonDeg[p][t][0];
      j=nonDeg[p][t][1];
      Zind(num1+t) = X[p](j,i)-X[p](i,j);
      Zind(num1+t) /= (eps[p][j]-eps[p][i]);;
    }
    num1 += nonDeg[p].size();
  }

  arma::vec Xov(nIA);
  num1=0;
  for(int p=0;p<D1.size();p++)
  {
    for(int a=0;a<nVirt[p];a++)
    {
      for(int i=0;i<nOcc[p];i++)
      {
	Xov(num1+a*nOcc[p]+i) = X[p](i,a+nOcc[p])-X[p](a+nOcc[p],i);
      }
    }
    num1 += nOcc[p]*nVirt[p];
  }

  arma::vec Zov = arma::solve(Aov,Xov-Aind*Zind);

  int nTot=nIA+nNonDeg;

  Z.set_size(nTot);
  Z.zeros();
  num1=0;
  num2=0;
  int numT=0;
  for(int p=0;p<D1.size();p++)
  {
    for(int t=0;t<nonDeg[p].size();t++)
    {
      i=nonDeg[p][t][0];
      j=nonDeg[p][t][1];
      Z(numT+t) = Zind(num1+t);
    }
    numT += nonDeg[p].size();
    num1 += nonDeg[p].size();
    for(int a=0;a<nVirt[p];a++)
    {
      for(int i=0;i<nOcc[p];i++)
      {
	Z(numT+a*nOcc[p]+i) = Zov(num2+a*nOcc[p]+i);
      }
    }
    numT += nOcc[p]*nVirt[p];
    num2 += nOcc[p]*nVirt[p];
  } 

  nTot=0;
  for(int p=0;p<D1.size();p++)
  {
    nTot += nMO[p]*(nMO[p]+1)/2;
  }

  dX.set_size(nTot);
  num1=0;
  for(int p=0;p<D1.size();p++)
  {
    for(int i=0;i<nMO[p];i++)
    {
      dX(num1) = 0.5*X[p](i,i);
      num1++;
      for(int j=0;j<i;j++)
      {
	dX(num1) = X[p](i,j);
	num1++;
      }
    }
  }

  return;
}

void findLabel(std::string label,std::vector<std::string> tLabel,int &nT)
{
  int n=0;
  while(n<tLabel.size() && tLabel[n] != label)
  {
    n++;
  }

  if(n < tLabel.size())
  {
    nT = n;
  }

  return;
}

void genIntsDer(double &dEnuc,std::vector<libint2::Atom> &atomsUnique,std::string basis,std::vector<arma::mat> &Ss,std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V,const std::vector<std::vector<std::vector<orbPair> > > &ao2mo,const std::vector<std::vector<std::vector<int> > > &bas,std::vector<int> nAOs,arma::Mat<int> prodTable,std::string pointGroup,std::vector<int> mode,double dx)
{
  double r1;
  double Enuc=0.0;
  dEnuc=0.0;

  int start,end;

  atomStartEnd(pointGroup,atomsUnique,mode[0],start,end);
  

  double *var;
  switch(mode[1])
  {
    case 0:
      var = &atomsUnique[mode[0]].x;
      break;
    case 1:
      var = &atomsUnique[mode[0]].y;
      break;
    case 2:
      var = &atomsUnique[mode[0]].z;
      break;
  }

  std::vector<libint2::Atom> atomsP,atomsM;

  (*var) += dx;
  genAtoms(basis,pointGroup,atomsUnique,atomsP);
  (*var) -= 2.0*dx;
  genAtoms(basis,pointGroup,atomsUnique,atomsM);
  (*var) += dx;
  
  double dxR = 0.5/dx;

  for(int i=0;i<atomsP.size();i++)
  {
    for(int j=i+1;j<atomsP.size();j++)
    {
      r1=(atomsP[i].x-atomsP[j].x)*(atomsP[i].x-atomsP[j].x);
      r1+=(atomsP[i].y-atomsP[j].y)*(atomsP[i].y-atomsP[j].y);
      r1+=(atomsP[i].z-atomsP[j].z)*(atomsP[i].z-atomsP[j].z);
      Enuc += (double) atomsP[i].atomic_number*atomsP[j].atomic_number/sqrt(r1);
    }
  }
  dEnuc += Enuc;

  Enuc=0.0;
  for(int i=0;i<atomsM.size();i++)
  {
    for(int j=i+1;j<atomsM.size();j++)
    {
      r1=(atomsM[i].x-atomsM[j].x)*(atomsM[i].x-atomsM[j].x);
      r1+=(atomsM[i].y-atomsM[j].y)*(atomsM[i].y-atomsM[j].y);
      r1+=(atomsM[i].z-atomsM[j].z)*(atomsM[i].z-atomsM[j].z);
      Enuc += (double) atomsM[i].atomic_number*atomsM[j].atomic_number/sqrt(r1);
    }
  }
  dEnuc -= Enuc;
  dEnuc *= dxR;

  libint2::BasisSet obsP(basis,atomsP);
  libint2::BasisSet obsM(basis,atomsM);

  libint2::init();

  libint2::OneBodyEngine engineSp(libint2::OneBodyEngine::overlap,obsP.max_nprim(),obsP.max_l(),0);
  libint2::OneBodyEngine engineKp(libint2::OneBodyEngine::kinetic,obsP.max_nprim(),obsP.max_l(),0);
  libint2::OneBodyEngine engineNp(libint2::OneBodyEngine::nuclear,obsP.max_nprim(),obsP.max_l(),0);
  libint2::OneBodyEngine engineSm(libint2::OneBodyEngine::overlap,obsM.max_nprim(),obsM.max_l(),0);
  libint2::OneBodyEngine engineKm(libint2::OneBodyEngine::kinetic,obsM.max_nprim(),obsM.max_l(),0);
  libint2::OneBodyEngine engineNm(libint2::OneBodyEngine::nuclear,obsM.max_nprim(),obsM.max_l(),0);


  std::vector<std::pair<double,std::array<double,3>>> qP;
  std::vector<std::pair<double,std::array<double,3>>> qM;
  for(int a=0;a<atomsP.size();a++) 
  {
    qP.push_back( {static_cast<double>(atomsP[a].atomic_number), {{atomsP[a].x, atomsP[a].y, atomsP[a].z}}} );
    qM.push_back( {static_cast<double>(atomsM[a].atomic_number), {{atomsM[a].x, atomsM[a].y, atomsM[a].z}}} );
  }
  engineNp.set_params(qP);
  engineNm.set_params(qM);

  auto shell2bf = obsP.shell2bf();
  auto shell2atom = obsP.shell2atom(atomsP);

  int num1,num2,num3,num4;
  double w1,w2,w3,w4;

  double deg;

  std::vector<double> intsS,intsK,intsN;

  Ss.resize(nAOs.size());
  K1s.resize(nAOs.size());
  for(int r=0;r<nAOs.size();r++)
  {
    Ss[r].resize(nAOs[r],nAOs[r]);
    Ss[r].zeros();
    K1s[r].resize(nAOs[r],nAOs[r]);
    K1s[r].zeros();
  }
  for(int s1=0;s1<obsP.size();s1++)
  {
    for(int s2=0;s2<=s1;s2++)
    {
      const auto* intsNp = engineNp.compute(obsP[s1],obsP[s2]);

      const auto* intsNm = engineNm.compute(obsM[s1],obsM[s2]);

      int bf1 = shell2bf[s1];
      int n1 = obsP[s1].size();
      int bf2 = shell2bf[s2];
      int n2 = obsP[s2].size();

      intsN.resize(n1*n2);

      for(int f1=0;f1<n1;f1++)
      {
	for(int f2=0;f2<n2;f2++)
	{
	  intsN[f1*n2+f2] = (intsNp[f1*n2+f2]-intsNm[f1*n2+f2])*dxR;
	}
      }

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
		K1s[r](num1,num2) += deg*w1*w2*intsN[f1*n2+f2];
		K1s[r](num2,num1) += deg*w1*w2*intsN[f1*n2+f2];
	      }
	    }
	  }
	}
      }
    }
  }

  bool calc=false;
  for(int s1=0;s1<obsP.size();s1++)
  {
    for(int s2=0;s2<=s1;s2++)
    {
int a1 = shell2atom[s1];
int a2 = shell2atom[s2];
if(a1 >= start && a1 <= end)
{
  calc=true;
}
if(a2 >= start && a2 <= end)
{
  calc=true;
}
if(calc)
{
      const auto* intsSp = engineSp.compute(obsP[s1],obsP[s2]);
      const auto* intsKp = engineKp.compute(obsP[s1],obsP[s2]);
//      const auto* intsNp = engineNp.compute(obsP[s1],obsP[s2]);

      const auto* intsSm = engineSm.compute(obsM[s1],obsM[s2]);
      const auto* intsKm = engineKm.compute(obsM[s1],obsM[s2]);
//      const auto* intsNm = engineNm.compute(obsM[s1],obsM[s2]);

      int bf1 = shell2bf[s1];
      int n1 = obsP[s1].size();
      int bf2 = shell2bf[s2];
      int n2 = obsP[s2].size();


      intsS.resize(n1*n2);
//      intsN.resize(n1*n2);
      intsK.resize(n1*n2);

      for(int f1=0;f1<n1;f1++)
      {
	for(int f2=0;f2<n2;f2++)
	{
	  intsS[f1*n2+f2] = (intsSp[f1*n2+f2]-intsSm[f1*n2+f2])*dxR;
//	  intsN[f1*n2+f2] = (intsNp[f1*n2+f2]-intsNm[f1*n2+f2])*dxR;
	  intsK[f1*n2+f2] = (intsKp[f1*n2+f2]-intsKm[f1*n2+f2])*dxR;
	}
      }

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
//		K1s[r](num1,num2) += deg*w1*w2*intsN[f1*n2+f2];
//		K1s[r](num2,num1) += deg*w1*w2*intsN[f1*n2+f2];
	      }
	    }
	  }
	}
      }

}
    }
  }

  libint2::TwoBodyEngine<libint2::Coulomb> engineVp(obsP.max_nprim(),obsP.max_l(), 0);
  libint2::TwoBodyEngine<libint2::Coulomb> engineVm(obsM.max_nprim(),obsM.max_l(), 0);

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
  for(int I=0;I<obsP.size();I++)
  {
    int bf1=shell2bf[I];
    int n1=obsP[I].size();
    for(int K=0;K<=I;K++)
    {
      int bf3=shell2bf[K];
      int n3=obsP[K].size();
      for(int J=0;J<obsP.size();J++)
      {
	int bf2=shell2bf[J];
	int n2=obsP[J].size();
	for(int L=0;L<=J && J*obsP.size()+L<=I*obsP.size()+K;L++)
	{
	  int bf4=shell2bf[L];
	  int n4=obsP[L].size();

	  int a1=shell2atom[I];
	  int a2=shell2atom[K];
	  int a3=shell2atom[J];
	  int a4=shell2atom[L];
	  calc=false;
	  if(a1 >= start && a1 <= end)
	  {
	    calc=true;
	  }
	  if(a2 >= start && a2 <= end)
	  {
	    calc=true;
	  }
	  if(a3 >= start && a3 <= end)
	  {
	    calc=true;
	  }
	  if(a4 >= start && a4 <= end)
	  {
	    calc=true;
	  }
	  if(calc)
	  { 
	  const auto* intsVp = engineVp.compute(obsP[I],obsP[K],obsP[J],obsP[L]);
	  const auto* intsVm = engineVm.compute(obsM[I],obsM[K],obsM[J],obsM[L]);
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
		  x = (intsVp[f1*n3*n2*n4+f3*n2*n4+f2*n4+f4]-intsVm[f1*n3*n2*n4+f3*n2*n4+f2*n4+f4])*dxR*deg;
		  //x = deg*ints_shellset[f1*n3*n2*n4+f3*n2*n4+f2*n4+f4];
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
  }


  return;
}

void readInputFile(char *fileName,char *xyzName,char *punchName,std::string &basis,std::string &pointGroup,int &method)
{
  std::string str; char c; int n;
  std::ifstream theFile;
  theFile.open(fileName);

  if(theFile.fail())
  {
    std::cerr << "Could not open inputFile" << std::endl;
  }
  else
  {
    theFile >> str;
    while(!theFile.eof())
    {
      if(str == "BASIS")
      {
	theFile >> c; theFile >> basis;
      }
      else if(str == "POINTGROUP")
      {
	theFile >> c; theFile >> pointGroup;
      }
      else if(str == "METHOD")
      {
	theFile >> c; theFile >> str;
	if(str == "z")
	{
	  method=2;
	}
	if(str == "u")
	{
	  method=1;
	}
	if(str == "num")
	{
	  method=0;
	}
      }
      else if(str == "MOLECULE")
      {
	theFile >> c; theFile >> str;
	std::strcpy(xyzName,str.c_str());
      }
      else if(str == "PUNCH")
      {
	theFile >> c; theFile >> str;
	std::strcpy(punchName,str.c_str());
      }
      theFile >> str;
    }
  }

  return;
}
