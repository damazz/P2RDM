#include "HF.h"
#include <chrono>
#include <iomanip>

void buildD1sym(const std::vector<int> ns,const std::vector<arma::mat> &Cs,std::vector<arma::mat> &D1s)
{
  for(int r=0;r<ns.size();r++)
  {
    D1s[r].zeros();
    for(int i=0;i<ns[r];i++)
    {
      D1s[r]+=kron(Cs[r].col(i),Cs[r].col(i).t());
    }
  }

  return;
}

void buildGsym(std::vector<arma::mat> &Gs,const std::vector<arma::mat> &D1s,std::vector<std::vector<arma::mat> > &W)
{
  double x,y;

  for(int p=0;p<Gs.size();p++)
  {
    arma::vec M(Gs[p].n_rows*Gs[p].n_rows,arma::fill::zeros);
    for(int r=0;r<Gs.size();r++)
    {
      if(Gs[r].n_rows > 0)
      {
	M += W[p][r]*arma::vectorise(D1s[r]);
      }
    }
    for(int i=0;i<Gs[p].n_rows;i++)
    {
      for(int k=0;k<Gs[p].n_rows;k++)
      {
	Gs[p](k,i) = M(i*Gs[p].n_rows+k);
      }
    }
  }

  return;
}

double solveHFsym(int nels,std::vector<arma::mat> &Cs,std::vector<arma::mat> &Ss,std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V2f,const std::vector<std::vector<std::vector<int> > > &bas,std::vector<int> &ns,std::vector<int> &nCore,std::vector<int> nAOs,bool print,int core,bool useOld)
{
  int nStep,n;
  double Enew,Eold,dE,dConv;
  double x;
  arma::mat U,X;
  arma::vec eigval;
  int maxDIIS=100;
  bool diis=false;

  std::vector<std::vector<arma::mat> > W(nAOs.size());
  for(int r=0;r<W.size();r++)
  {
    for(int s=0;s<W.size();s++)
    {
      W[r].push_back(arma::mat(nAOs[r]*nAOs[r],nAOs[s]*nAOs[s],arma::fill::zeros));
    }
  }
  for(int p=0;p<bas.size();p++)
  {
    for(int q=0;q<bas.size();q++)
    {
      for(int i=0;i<nAOs[p];i++)
      {
	for(int k=0;k<nAOs[p];k++)
	{
	  for(int j=0;j<nAOs[q];j++)
	  {
	    for(int l=0;l<nAOs[q];l++)
	    {
	      W[q][p](j*nAOs[q]+l,i*nAOs[p]+k) = 2.0*V2f[p*bas.size()+q][p*bas.size()+q](k*nAOs[q]+l,i*nAOs[q]+j);
	      W[q][p](j*nAOs[q]+l,i*nAOs[p]+k)-= V2f[q*bas.size()+p][p*bas.size()+q](l*nAOs[p]+k,i*nAOs[q]+j);
	    }
	  }
	}
      }
    }
  }

  std::vector<std::vector<arma::mat> > priorFs,Es;

  std::vector<arma::mat> Xs;
  int nMO=0;
  for(int r=0;r<nAOs.size();r++)
  {
    eig_sym(eigval,U,Ss[r]);
    minTheta(Ss[r],arma::mat(Ss[r].n_rows,Ss[r].n_cols,arma::fill::eye),eigval,U,false,0);
    n=0;
    while(n < eigval.n_rows && eigval(n) < 0.00001)
    {
      n++;
    }
    X.resize(eigval.n_rows,nAOs[r]-n);
    for(int i=0;i<X.n_cols;i++)
    {
      X.col(X.n_cols-i-1)=U.col(U.n_cols-i-1)/sqrt(eigval(U.n_cols-i-1));
    }
    Xs.push_back(X);
    nMO += X.n_cols;
  }

  std::vector<arma::mat> Cold;
  std::vector<arma::mat> D1s(K1s.size());
  std::vector<arma::mat> D1sOld(K1s.size());
  std::vector<arma::mat> Gs(K1s.size());

  for(int r=0;r<Ss.size();r++)
  {
    D1s[r].resize(nAOs[r],nAOs[r]); D1s[r].zeros();
    Gs[r].resize(nAOs[r],nAOs[r]); Gs[r].zeros();
  }

  bool matchC=false;
  if(useOld)
  {
    matchC=true;
    for(int r=0;r<ns.size();r++)
    {
      ns[r] += nCore[r];
    }
    Cold=Cs;
    buildD1sym(ns,Cs,D1s);
    buildGsym(Gs,D1s,W);
  }
  else
  {
    for(int r=0;r<Ss.size();r++)
    {
      Cold.push_back(arma::mat(Ss[r].n_rows,Ss[r].n_rows,arma::fill::eye));
      Cold[r] = Xs[r]*Cold[r];
    }
  }

  if(print)
  {
    std::cout << "nMO: " << nMO << std::endl;
  }

  arma::mat B;

  std::vector<arma::mat> Fs(Xs.size());
  std::vector<arma::vec> eigs(Xs.size());
  int max;

  dConv=1.0; dE=1.0; nStep=0;
  Enew=0.0;
  if(print)
  {
    std::cout << "Iter\tElec. E\t\t\tdE\t\t\tdConv" << std::endl;
  }
  diis=true;
  while(fabs(dE) > pow(10,-12) || dConv > 0.000001)
  {
    nStep++;
    for(int r=0;r<Cs.size();r++)
    {
      Fs[r]=K1s[r]+Gs[r];
    }

    if(diis)
    {
      DIIS(Fs,D1s,Ss,priorFs,Es,B,maxDIIS);    
    }

    for(int r=0;r<Cs.size();r++)
    {
      Fs[r]=Xs[r].t()*Fs[r]*Xs[r];
      eig_sym(eigs[r],Cs[r],Fs[r]);
      Cs[r]=Xs[r]*Cs[r];
      ns[r]=0;
    }

    for(int i=0;i<nels/2;i++)
    {
      max=-1; x=10000000;
      for(int r=0;r<Xs.size();r++)
      {
	if(ns[r]<eigs[r].n_rows && eigs[r](ns[r])<x)
	{
	  max=r;
	  x=eigs[r](ns[r]);
	}
      }
      ns[max] += 1;
    }
    D1sOld = D1s;
    buildD1sym(ns,Cs,D1s);

    buildGsym(Gs,D1s,W);
    Eold=Enew;
    Enew=0.0;
    double E1=0.0;
    double E2=0.0;
    for(int r=0;r<Gs.size();r++)
    {
      Enew += 2.0*trace(K1s[r]*D1s[r])+trace(Gs[r]*D1s[r]);
      E1 += 2.0*trace(K1s[r]*D1s[r]);
      E2 += trace(Gs[r]*D1s[r]);
    }

    dE = Enew-Eold;
/*    if(!diis && dE < 0.0 && dE > -0.001)
    {
      diis=true;
      if(print)
      {
	std::cout << "Turning on DIIS" << std::endl;
      }
    }
    else if(!diis && dE > 0.0)
    {
      for(int r=0;r<D1s.size();r++)
      {
//	D1s[r] = 0.5*(D1s[r]+D1sOld[r]);
      }
    }
*/    dConv=0.0;
    for(int r=0;r<D1s.size();r++)
    {
      dConv += arma::norm(D1s[r]-D1sOld[r]);
    }
    if(print)
    {
      printf("%d\t%.12f\t%.12E\t%.12E\n",nStep,Enew,dE,dConv);
      //printf("%d\t%.12f\t%.12E\t%.12E\n",nStep,Enew,dE,dConv);
    }
  }

  for(int r=0;r<Cs.size();r++)
  {
    minTheta(Ss[r],Cold[r],eigs[r],Cs[r],matchC,ns[r]);
  }

  nCore.resize(ns.size());
  for(int r=0;r<nCore.size();r++)
  {
    nCore[r]=0;
  }
  for(int i=0;i<core;i++)
  {
    max=-1; x=10000000;
    for(int r=0;r<Xs.size();r++)
    {
      if(nCore[r]<eigs[r].n_rows && eigs[r](nCore[r])<x)
      {
	max=r;
	x=eigs[r](nCore[r]);
      }
    }
    nCore[max] += 1;
    ns[max] -= 1;
  }

  return Enew;
}

void DIIS(std::vector<arma::mat> &Fn,const std::vector<arma::mat> &D1,const std::vector<arma::mat> &S,std::vector<std::vector<arma::mat> > &Fs,std::vector<std::vector<arma::mat> > &Es,arma::mat &B,int nF)
{
  std::vector<arma::mat> En;
  double x=0.0;
  for(int r=0;r<D1.size();r++)
  {
    x += arma::trace(D1[r]);
  }
  if(x < 0.000001)
  {
    return;
  }
  if(Es.size() == 0)
  {
    Fs.push_back(Fn);
    En.clear();
    x=0.0;
    for(int r=0;r<D1.size();r++)
    {
      En.push_back(Fn[r]*D1[r]*S[r]-S[r]*D1[r]*Fn[r]);
      x += arma::dot(En[r],En[r]);
    }
    Es.push_back(En);
    B.resize(2,2);
    B(0,0) = x;
    B(1,0)=B(0,1)=-1;
  }
  else if(Es.size() < nF)
  {
    Fs.push_back(Fn);
    En.clear();
    x=0.0;
    for(int r=0;r<D1.size();r++)
    {
      En.push_back(Fn[r]*D1[r]*S[r]-S[r]*D1[r]*Fn[r]);
      x += arma::dot(En[r],En[r]);
    }
    Es.push_back(En);
    B.resize(Fs.size()+1,Fs.size()+1);
    B(B.n_rows-2,B.n_cols-2)=x;
    for(int i=0;i<B.n_rows-2;i++)
    {
      x=0.0;
      for(int r=0;r<D1.size();r++)
      {
	x += arma::dot(Es[i][r],Es[Es.size()-1][r]);
      }
      B(i,Es.size()-1)=B(Es.size()-1,i)=x;
      B(i,Es.size())=B(Es.size(),i)=-1.0;
    }
    B(Es.size(),Es.size())=0.0;
    B(B.n_rows-1,B.n_rows-2)=B(B.n_rows-2,B.n_cols-1)=-1;
//    B.print();
  }
  else
  {
    for(int i=0;i<Es.size()-1;i++)
    {
      Es[i]=Es[i+1];
      Fs[i]=Fs[i+1];
    }
    B.submat(0,0,nF-2,nF-2)=B.submat(1,1,nF-1,nF-1);
    En.clear();
    x=0.0;
    for(int r=0;r<D1.size();r++)
    {
      En.push_back(Fn[r]*D1[r]*S[r]-S[r]*D1[r]*Fn[r]);
      x += arma::dot(En[r],En[r]);
    }
    B(B.n_rows-2,B.n_cols-2)=x;

    Es[Es.size()-1]=En;
    Fs[Fs.size()-1]=Fn;
    for(int i=0;i<B.n_rows-2;i++)
    {
      x=0.0;
      for(int r=0;r<D1.size();r++)
      {
	x += arma::dot(Es[i][r],Es[Es.size()-1][r]);
      }
      B(i,Es.size()-1)=B(Es.size()-1,i)=x;
      B(i,Es.size())=B(Es.size(),i)=-1.0;
    }
    B(Es.size(),Es.size())=0.0;
    B(B.n_rows-1,B.n_rows-2)=B(B.n_rows-2,B.n_cols-1)=-1;
  }

  arma::vec RHS(Es.size()+1,arma::fill::zeros);
  RHS(Es.size())=-1.0;

  arma::vec C = solve(B,RHS);

  Fn.clear();
  arma::mat F;
  for(int r=0;r<D1.size();r++)
  {
    F.set_size(D1[r].n_rows,D1[r].n_cols); F.zeros();
    for(int i=0;i<Es.size();i++)
    {
      F += C(i)*Fs[i][r];
    }
    Fn.push_back(F);
  }
  return;
}

void rotateInts(std::vector<arma::mat> &K1s,std::vector<std::vector<arma::mat> > &V2s,const std::vector<arma::mat> &Cs,std::vector<std::vector<std::vector<int> > > bas,std::vector<int> nOcc,std::vector<int> nCore,double &Ecore)
{
  for(int r=0;r<K1s.size();r++)
  {
    K1s[r] = Cs[r].t()*K1s[r]*Cs[r];
  }

  int p,q,r,s;
  int nRow,nCol;
  int n1,n2;
  for(int t=0;t<bas.size();t++)
  {
    nRow=0;
    for(int m=0;m<bas[t].size();m++)
    {
      p=bas[t][m][0];
      q=bas[t][m][1];
      n1=Cs[p].n_cols*Cs[q].n_cols;

      nCol=0;
      for(int n=0;n<bas[t].size();n++)
      {
	r=bas[t][n][0];
	s=bas[t][n][1];
	n2=Cs[r].n_cols*Cs[s].n_cols;

	if(n1>0 && n2>0)
	{
	  V2s[p*bas.size()+q][r*bas.size()+s] = kron(Cs[p].t(),Cs[q].t())*V2s[p*bas.size()+q][r*bas.size()+s]*kron(Cs[r],Cs[s]);
	}

	nCol += n2;
      }

      nRow += n1;
    }
  }

/*  int nels=0;
  for(int i=0;i<nOcc.size();i++)
  {
    nels += nOcc[i];
  }
  int nMO=0;
  for(int r=0;r<Cs.size();r++)
  {
    nMO += Cs[r].n_cols-nCore[r];
  }

  std::vector<int> nVirt;
  for(int i=0;i<nOcc.size();i++)
  {
    nVirt.push_back(Cs[i].n_cols-nOcc[i]-nCore[i]);
  }

  K1.set_size(nMO,nMO);
  K1.zeros();

  int row1,row2,col1,col2;
  nRow=0; nCol=0;
  for(int p=0;p<K1s.size();p++)
  {
    for(int n1=0;n1<nOcc[p];n1++)
    {
      for(int n2=0;n2<nOcc[p];n2++)
      {
	K1(nRow+n2,nRow+n1) = K1s[p](n2+nCore[p],n1+nCore[p]);
	for(int q=0;q<K1s.size();q++)
	{
	  for(int n3=0;n3<nCore[q];n3++)
	  {
	    K1(nRow+n2,nRow+n1) += 2.0*V2s[q*K1s.size()+p][q*K1s.size()+p](n3*Cs[p].n_cols+n2+nCore[p],n3*Cs[p].n_cols+n1+nCore[p]);
	    K1(nRow+n2,nRow+n1) -= V2s[p*K1s.size()+q][q*K1s.size()+p]((n2+nCore[p])*Cs[q].n_cols+n3,n3*Cs[p].n_cols+n1+nCore[p]);
	  }
	}
      }
      for(int n2=0;n2<nVirt[p];n2++)
      {
	K1(nCol+n2+nels,nRow+n1) = K1s[p](n2+nOcc[p]+nCore[p],n1+nCore[p]);
	for(int q=0;q<K1s.size();q++)
	{
	  for(int n3=0;n3<nCore[q];n3++)
	  {
	    K1(nCol+n2+nels,nRow+n1) += 2.0*V2s[q*K1s.size()+p][q*K1s.size()+p](n3*Cs[p].n_cols+n2+nOcc[p]+nCore[p],n3*Cs[p].n_cols+n1+nCore[p]);
	    K1(nCol+n2+nels,nRow+n1) -= V2s[p*K1s.size()+q][q*K1s.size()+p]((n2+nOcc[p]+nCore[p])*Cs[q].n_cols+n3,n3*Cs[p].n_cols+n1+nCore[p]);
	  }
	}
      }
    }
    for(int n1=0;n1<nVirt[p];n1++)
    {
      for(int n2=0;n2<nOcc[p];n2++)
      {
	K1(nRow+n2,nCol+n1+nels) = K1s[p](n2+nCore[p],n1+nOcc[p]+nCore[p]);
	for(int q=0;q<K1s.size();q++)
	{
	  for(int n3=0;n3<nCore[q];n3++)
	  {
	    K1(nRow+n2,nCol+n1+nels) += 2.0*V2s[q*K1s.size()+p][q*K1s.size()+p](n3*Cs[p].n_cols+n2+nCore[p],n3*Cs[p].n_cols+n1+nCore[p]+nOcc[p]);
	    K1(nRow+n2,nCol+n1+nels) -= V2s[p*K1s.size()+q][q*K1s.size()+p]((n2+nCore[p])*Cs[q].n_cols+n3,n3*Cs[p].n_cols+n1+nCore[p]+nOcc[p]);
	  }
	}
      }
      for(int n2=0;n2<nVirt[p];n2++)
      {
	K1(nCol+n2+nels,nCol+n1+nels) = K1s[p](n2+nOcc[p]+nCore[p],n1+nOcc[p]+nCore[p]);
	for(int q=0;q<K1s.size();q++)
	{
	  for(int n3=0;n3<nCore[q];n3++)
	  {
	    K1(nCol+n2+nels,nCol+n1+nels) += 2.0*V2s[q*K1s.size()+p][q*K1s.size()+p](n3*Cs[p].n_cols+n2+nOcc[p]+nCore[p],n3*Cs[p].n_cols+n1+nCore[p]+nOcc[p]);
	    K1(nCol+n2+nels,nCol+n1+nels) -= V2s[p*K1s.size()+q][q*K1s.size()+p]((n2+nOcc[p]+nCore[p])*Cs[q].n_cols+n3,n3*Cs[p].n_cols+n1+nCore[p]+nOcc[p]);
	  }
	}
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
		    V2ab((o3+n3)*nMO+o4+n4,(o1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nCore[r])*Cs[s].n_cols+n4+nCore[s],(n1+nCore[p])*Cs[q].n_cols+n2+nCore[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+v4+n4,(o1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nCore[r])*Cs[s].n_cols+n4+nOcc[s]+nCore[s],(n1+nCore[p])*Cs[q].n_cols+n2+nCore[q]);
		  }
		}
		for(int n3=0;n3<nVirt[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+o4+n4,(o1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r]+nCore[r])*Cs[s].n_cols+n4+nCore[s],(n1+nCore[p])*Cs[q].n_cols+n2+nCore[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+v4+n4,(o1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r]+nCore[r])*Cs[s].n_cols+n4+nOcc[s]+nCore[s],(n1+nCore[p])*Cs[q].n_cols+n2+nCore[q]);
		  }
		}
	      }
	      for(int n2=0;n2<nVirt[q];n2++)
	      {
		for(int n3=0;n3<nOcc[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+o4+n4,(o1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nCore[r])*Cs[s].n_cols+n4+nCore[s],(n1+nCore[p])*Cs[q].n_cols+n2+nOcc[q]+nCore[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+v4+n4,(o1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nCore[r])*Cs[s].n_cols+n4+nOcc[s]+nCore[s],(n1+nCore[p])*Cs[q].n_cols+n2+nOcc[q]+nCore[q]);
		  }
		}
		for(int n3=0;n3<nVirt[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+o4+n4,(o1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r]+nCore[r])*Cs[s].n_cols+n4+nCore[s],(n1+nCore[p])*Cs[q].n_cols+n2+nOcc[q]+nCore[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+v4+n4,(o1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r]+nCore[r])*Cs[s].n_cols+n4+nOcc[s]+nCore[s],(n1+nCore[p])*Cs[q].n_cols+n2+nOcc[q]+nCore[q]);
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
		    V2ab((o3+n3)*nMO+o4+n4,(v1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nCore[r])*Cs[s].n_cols+n4+nCore[s],(n1+nOcc[p]+nCore[p])*Cs[q].n_cols+n2+nCore[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+v4+n4,(v1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nCore[r])*Cs[s].n_cols+n4+nOcc[s]+nCore[s],(n1+nOcc[p]+nCore[p])*Cs[q].n_cols+n2+nCore[q]);
		  }
		}
		for(int n3=0;n3<nVirt[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+o4+n4,(v1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r]+nCore[r])*Cs[s].n_cols+n4+nCore[s],(n1+nOcc[p]+nCore[p])*Cs[q].n_cols+n2+nCore[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+v4+n4,(v1+n1)*nMO+o2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r]+nCore[r])*Cs[s].n_cols+n4+nOcc[s]+nCore[s],(n1+nOcc[p]+nCore[p])*Cs[q].n_cols+n2+nCore[q]);
		  }
		}
	      }
	      for(int n2=0;n2<nVirt[q];n2++)
	      {
		for(int n3=0;n3<nOcc[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+o4+n4,(v1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nCore[r])*Cs[s].n_cols+n4+nCore[s],(n1+nOcc[p]+nCore[p])*Cs[q].n_cols+n2+nOcc[q]+nCore[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((o3+n3)*nMO+v4+n4,(v1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nCore[r])*Cs[s].n_cols+n4+nOcc[s]+nCore[s],(n1+nOcc[p]+nCore[p])*Cs[q].n_cols+n2+nOcc[q]+nCore[q]);
		  }
		}
		for(int n3=0;n3<nVirt[r];n3++)
		{
		  for(int n4=0;n4<nOcc[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+o4+n4,(v1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r]+nCore[r])*Cs[s].n_cols+n4+nCore[s],(n1+nOcc[p]+nCore[p])*Cs[q].n_cols+n2+nOcc[q]+nCore[q]);
		  }
		  for(int n4=0;n4<nVirt[s];n4++)
		  {
		    V2ab((v3+n3)*nMO+v4+n4,(v1+n1)*nMO+v2+n2) = V2s[r*bas.size()+s][p*bas.size()+q]((n3+nOcc[r]+nCore[r])*Cs[s].n_cols+n4+nOcc[s]+nCore[s],(n1+nOcc[p]+nCore[p])*Cs[q].n_cols+n2+nOcc[q]+nCore[q]);
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
*/
  Ecore=0.0;
  for(int p=0;p<nCore.size();p++)
  {
    for(int i=0;i<nCore[p];i++)
    {
      Ecore += 2.0*K1s[p](i,i);
      for(int q=0;q<nCore.size();q++)
      {
	for(int j=0;j<nCore[q];j++)
	{
	  Ecore += 2.0*V2s[p*nCore.size()+q][p*nCore.size()+q](i*Cs[q].n_cols+j,i*Cs[q].n_cols+j);
	  Ecore -= V2s[q*nCore.size()+p][p*nCore.size()+q](j*Cs[p].n_cols+i,i*Cs[q].n_cols+j);
	}
      }
    }
  }

  return;
}

void minTheta(const arma::mat &S,const arma::mat &Cold,const arma::vec &eigs,arma::mat &Cnew,bool flag,int nels)
{
  int start,end;
  int num;
  double max,theta,cosT,sinT;

  int virt=Cnew.n_cols-nels;

  double F,F2;
  arma::mat o,v,OV;
  arma::mat U(2,2);
  arma::vec v2,v3;

  double o1v1,o1v2,o2v1,o2v2;

  double overlap;
  for(int m=0;m<eigs.n_rows;m++)
  {
    start=m;
    end=start;
    while(end < eigs.n_rows-1 && eigs(end+1)-eigs(start) < 0.000001)
    {
      end++;
    }
    //if(end != start)
    {
      v = Cnew.cols(start,end);
      o.set_size(v.n_rows,v.n_cols);
      for(int i=0;i<v.n_cols;i++)
      {
	v2 = S*v.col(i);
	max=0.0; num=0;
	for(int j=0;j<v2.n_rows;j++)
	{
	  overlap = arma::dot(Cold.col(j),v2);
	  if(fabs(overlap) > fabs(max))
	  {
	    num=j;
	    max = overlap;
	  }
	}
	o.col(i) = Cold.col(num);
	if(max < 0.0)
	{
	  v.col(i) *= -1.0;
	}
      }
      max=1.0;
      while(max > 0.00000001)
      {
	max=0.0;
	for(int i=0;i<v.n_cols;i++)
	{
	  for(int j=i+1;j<v.n_cols;j++)
	  {
	    o1v1 = arma::dot(o.col(i),S*v.col(i));
	    o1v2 = arma::dot(o.col(i),S*v.col(j));
	    o2v2 = arma::dot(o.col(j),S*v.col(j));
	    o2v1 = arma::dot(o.col(j),S*v.col(i));

	    if(fabs(o1v2-o2v1)>max)
	    {
	      max = fabs(o1v2-o2v1);
	    }
      
	    theta = atan((o1v2-o2v1)/(o2v2+o1v1));
	    cosT = cos(theta);
	    sinT = sin(theta);

	    v2 = cosT*v.col(i)+sinT*v.col(j);
	    v3 = -sinT*v.col(i)+cosT*v.col(j);
	    v.col(i) = v2;
	    v.col(j) = v3;
	  }
	}
      }
      for(int i=0;i<v.n_cols;i++)
      {
	Cnew.col(start+i) = v.col(i);
      }
//Cnew.print();
//std::cout << std::endl;
    }
    m=end;
  }

  if(flag)
  {
    arma::mat Ctemp(Cnew.n_rows,Cnew.n_cols,arma::fill::zeros);
    for(int n=0;n<nels;n++)
    {
      v2 = S*Cold.col(n);
      num=0; max=0.0;
      for(int j=0;j<nels;j++)
      {
	overlap = arma::dot(Cnew.col(j),v2);
	if(fabs(overlap) > fabs(max))
	{
	  max=fabs(overlap);
	  num=j;
	}
      }
      Ctemp.col(n)=Cnew.col(num);
      Cnew.col(num).zeros();
    }

    for(int n=nels;n<Cnew.n_cols;n++)
    {
      v2 = S*Cold.col(n);
      num=0; max=0.0;
      for(int j=nels;j<Cnew.n_cols;j++)
      {
	overlap = arma::dot(Cnew.col(j),v2);
	if(fabs(overlap) > fabs(max))
	{
	  max=fabs(overlap);
	  num=j;
	}
      }
      Ctemp.col(n)=Cnew.col(num);
      Cnew.col(num).zeros();
    }

    Cnew=Ctemp;
  }

  return;
}
