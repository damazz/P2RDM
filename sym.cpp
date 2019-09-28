#include "sym.h"
#include <iomanip>

//I believe libint's d-orbital ordering is xy,yz,zz,xz,xx-yy for spherical harmonics
//for f-orbital ordering in spherical harmonics, the ordering must be (choose two in parentheses)
//	y(z2,y2,x2),xyz,y(z2,y2,x2),z(z2,y2,x2),x(z2,y2,x2),z(z2,y2,x2),x(z2,y2,x2)
//for g in spherical harmonics, I have no idea what the functional form is, but they transform as follows in D2h
//	B1g,B3g,B1g,B3g,Ag,B2g,Ag,B2g,Ag

void invert(libint2::Atom &a,arma::Row<int> &orb,int l,bool ispher)
{
  a.x *= -1.0;
  a.y *= -1.0;
  a.z *= -1.0;
  switch(l)
  {
    case 0:
      break;
    case 1:
      orb *= -1;
      break;
    case 2:
      break;
    case 3:
      orb *= -1;
      break;
    case 4:
      break;
  }

  return;
}

void c2(int axis,libint2::Atom &a,arma::Row<int> &orb,int l,bool ispher)
{
  if(axis==0) //c2x
  {
    a.y *= -1.0;
    a.z *= -1.0;
    switch (l)
    {
      case 0:
	break;
      case 1:
	orb(1) *= -1;
	orb(2) *= -1;
	break;
      case 2:
	if(ispher)
	{
	  orb(0) *= -1;
	  orb(3) *= -1;
	}
	else
	{
	  orb(1) *= -1;
	  orb(2) *= -1;
	}
	break;
      case 3:
	if(ispher)
	{
	  orb(0) *= -1.0;
	  orb(2) *= -1.0;
	  orb(3) *= -1.0;
	  orb(5) *= -1.0;
	}
	else
	{
	  orb(1) *= -1.0;
	  orb(2) *= -1.0;
	  orb(6) *= -1.0;
	  orb(7) *= -1.0;
	  orb(8) *= -1.0;
	  orb(9) *= -1.0;
	}
	break;
      case 4:
	if(ispher)
	{
	  orb(0) *= -1.0;
	  orb(2) *= -1.0;
	  orb(5) *= -1.0;
	  orb(7) *= -1.0;
	}
	else
	{
	  orb(1) *= -1.0;
	  orb(2) *= -1.0;
	  orb(6) *= -1.0;
	  orb(7) *= -1.0;
	  orb(8) *= -1.0;
	  orb(9) *= -1.0;
	}
	break;
    }
  }
  if(axis==1) //c2y
  {
    a.x *= -1.0;
    a.z *= -1.0;
    switch (l)
    {
      case 0:
	break;
      case 1:
	orb(0) *= -1;
	orb(2) *= -1;
	break;
      case 2:
	if(ispher)
	{
	  orb(0) *= -1;
	  orb(1) *= -1;
	}
	else
	{
	  orb(1) *= -1;
	  orb(4) *= -1;
	}
	break;
      case 3:
	if(ispher)
	{
	  orb(3) *= -1.0;
	  orb(4) *= -1.0;
	  orb(5) *= -1.0;
	  orb(6) *= -1.0;
	}
	else
	{
	  orb(0) *= -1.0;
	  orb(2) *= -1.0;
	  orb(3) *= -1.0;
	  orb(5) *= -1.0;
	  orb(7) *= -1.0;
	  orb(9) *= -1.0;
	}
	break;
      case 4:
	if(ispher)
	{
	  orb(0) *= -1.0;
	  orb(1) *= -1.0;
	  orb(2) *= -1.0;
	  orb(3) *= -1.0;
	}
	else
	{
	  orb(1) *= -1.0;
	  orb(4) *= -1.0;
	  orb(6) *= -1.0;
	  orb(8) *= -1.0;
	  orb(11) *= -1.0;
	  orb(13) *= -1.0;
	}
	break;
    }
  }
  if(axis==2) //c2z
  {
    a.x *= -1.0;
    a.y *= -1.0;
    switch (l)
    {
      case 0:
	break;
      case 1:
	orb(0) *= -1;
	orb(1) *= -1;
	break;
      case 2:
	if(ispher)
	{
	  orb(1) *= -1;
	  orb(3) *= -1;
	}
	else
	{
	  orb(2) *= -1;
	  orb(4) *= -1;
	}
	break;
      case 3:
	if(ispher)
	{
	  orb(0) *= -1.0;
	  orb(2) *= -1.0;
	  orb(4) *= -1.0;
	  orb(6) *= -1.0;
	}
	else
	{
	  orb(0) *= -1.0;
	  orb(1) *= -1.0;
	  orb(3) *= -1.0;
	  orb(5) *= -1.0;
	  orb(6) *= -1.0;
	  orb(8) *= -1.0;
	}
	break;
      case 4:
	if(ispher)
	{
	  orb(1) *= -1.0;
	  orb(3) *= -1.0;
	  orb(5) *= -1.0;
	  orb(7) *= -1.0;
	}
	else
	{
	  orb(2) *= -1.0;
	  orb(4) *= -1.0;
	  orb(7) *= -1.0;
	  orb(9) *= -1.0;
	  orb(11) *= -1.0;
	  orb(13) *= -1.0;
	}
	break;
    }
  }

  return;
}

void sig(int norm,libint2::Atom &a,arma::Row<int> &orb,int l,bool ispher)
{
  if(norm==0) //sig-yz
  {
    a.x *= -1.0;
    switch(l)
    {
      case 0:
	break;
      case 1:
	orb(0) *= -1;
	break;
      case 2:
	if(ispher)
	{
	  orb(0) *= -1;
	  orb(3) *= -1;
	}
	else
	{
	  orb(1) *= -1;
	  orb(2) *= -1;
	}
	break;
      case 3:
	if(ispher)
	{
	  orb(1) *= -1.0;
	  orb(4) *= -1.0;
	  orb(6) *= -1.0;
	}
	else
	{
	  orb(1) *= -1.0;
	  orb(2) *= -1.0;
	  orb(6) *= -1.0;
	  orb(7) *= -1.0;
	  orb(8) *= -1.0;
	  orb(9) *= -1.0;
	}
	break;
      case 4:
	if(ispher)
	{
	  orb(0) *= -1.0;
	  orb(2) *= -1.0;
	  orb(5) *= -1.0;
	  orb(7) *= -1.0;
	}
	else
	{
	  orb(1) *= -1.0;
	  orb(2) *= -1.0;
	  orb(6) *= -1.0;
	  orb(7) *= -1.0;
	  orb(8) *= -1.0;
	  orb(9) *= -1.0;
	}
	break;
    }
  }
  if(norm==1) //sig-xz
  {
    a.y *= -1.0;
    switch(l)
    {
      case 0:
	break;
      case 1:
	orb(1) *= -1;
	break;
      case 2:
	if(ispher)
	{
	  orb(0) *= -1;
	  orb(1) *= -1;
	}
	else
	{
	  orb(1) *= -1;
	  orb(4) *= -1;
	}
	break;
      case 3:
	if(ispher)
	{
	  orb(0) *= -1.0;
	  orb(1) *= -1.0;
	  orb(2) *= -1.0;
	}
	else
	{
	  orb(0) *= -1.0;
	  orb(2) *= -1.0;
	  orb(3) *= -1.0;
	  orb(5) *= -1.0;
	  orb(7) *= -1.0;
	  orb(9) *= -1.0;
	}
	break;
      case 4:
	if(ispher)
	{
	  orb(0) *= -1.0;
 	  orb(1) *= -1.0;
	  orb(2) *= -1.0;
	  orb(3) *= -1.0;
	}
	else
	{
	  orb(1) *= -1.0;
	  orb(4) *= -1.0;
	  orb(6) *= -1.0;
	  orb(8) *= -1.0;
	  orb(11) *= -1.0;
	  orb(13) *= -1.0;
	}
	break;
    }
  }
  if(norm==2) //sig-xy
  {
    a.z *= -1.0;
    switch(l)
    {
      case 0:
	break;
      case 1:
	orb(2) *= -1;
	break;
      case 2:
	if(ispher)
	{
	  orb(1) *= -1;
	  orb(3) *= -1;
	}
	else
	{
	  orb(2) *= -1;
	  orb(4) *= -1;
	}
	break;
      case 3:
	if(ispher)
	{
	  orb(1) *= -1.0;
	  orb(3) *= -1.0;
	  orb(5) *= -1.0;
	}
	else
	{
	  orb(0) *= -1.0;
	  orb(1) *= -1.0;
	  orb(3) *= -1.0;
	  orb(5) *= -1.0;
	  orb(6) *= -1.0;
	  orb(8) *= -1.0;
	}
	break;
      case 4:
	if(ispher)
	{
	  orb(1) *= -1.0;
	  orb(3) *= -1.0;
	  orb(5) *= -1.0;
	  orb(7) *= -1.0;
	}
	else
	{
	  orb(2) *= -1.0;
	  orb(4) *= -1.0;
	  orb(7) *= -1.0;
	  orb(9) *= -1.0;
	  orb(11) *= -1.0;
	  orb(13) *= -1.0;
	}
	break;
    }
  }

  return;
}

void reduce(arma::rowvec r,arma::mat c2vTable,arma::rowvec &groups)
{
  groups.resize(r.n_cols); groups.zeros();
  for(int i=0;i<c2vTable.n_rows;i++)
  {
    groups(i)=arma::dot(r,c2vTable.row(i));
  }

  return;
}

double symOp(libint2::Atom a1,libint2::Atom &a2,int l,int ml,std::string op,bool ispher)
{
  double x;
  int numL;
  if(ispher)
  {
    numL=2*l+1;
  }
  else
  {
    numL=l+1+l*(l+1)/2;
  }
  arma::Row<int> orb1(numL,arma::fill::zeros);
  arma::Row<int> orb2(numL,arma::fill::zeros);

  orb1(ml)=orb2(ml)=1;
  int n;

  if(op[0]=='e')
  {
    a2=a2;
    orb2=orb2;
  }
  else if(op[0]=='c')
  {
    if(op[2]=='x')
    {
      n=0;
    }
    else if(op[2]=='y')
    {
      n=1;
    }
    else if(op[2]=='z')
    {
      n=2;
    }
    c2(n,a2,orb2,l,ispher);
  }
  else if(op[0]=='v')
  {
    if(op[1]=='x' && op[2]=='y')
    {
      n=2;
    }
    else if(op[1]=='x' && op[2]=='z')
    {
      n=1;
    }
    else if(op[1]=='y' && op[2]=='z')
    {
      n=0;
    }
    sig(n,a2,orb2,l,ispher);
  }
  else if(op[0]=='i')
  {
    invert(a2,orb2,l,ispher);
  }

  if(a2.x==a1.x && a2.y==a1.y && a2.z==a1.z)
  {
    return dot(orb1,orb2);
  }
  else
  {
    return 0.0;
  }
}

void applySym(std::vector<libint2::Atom> atoms,std::string basis,std::string pointGroup,std::vector<libint2::Atom> &allAtoms,std::vector<arma::mat> &Cs,std::vector<std::string> &irreps,int &nMO)
{
  std::vector<std::string> ops;
  arma::mat charTable;
  //initPG(pointGroup,irreps,ops,charTable);

  libint2::BasisSet obs(basis,atoms);
  auto shell2atom=obs.shell2atom(atoms);
  std::vector<int> nMOs;
  int l;
  bool ispher=false;
  for(int i=0;i<basis.size()-1;i++)
  {
    if(basis[i]=='c' && basis[i+1]=='c')
    {
      ispher=true;
    }
  }

  for(int n=0;n<obs.size();n++)
  {
    l=obs[n].contr[0].l;
    if(shell2atom[n]+1 > nMOs.size())
    {
      if(ispher)
      {
	nMOs.push_back(2*l+1);
      }
      else
      {
	nMOs.push_back(l+1+l*(l+1)/2);
      }
    }
    else
    {
      if(ispher)
      {
	nMOs[shell2atom[n]] += 2*l+1;
      }
      else
      {
	nMOs[shell2atom[n]] += l+1+l*(l+1)/2;
      }
    }
  }

  libint2::Atom atom1;
 
  int num;
  bool found;
  std::vector<int> numIdent;
  nMO=0;
  for(int n=0;n<atoms.size();n++)
  {
    num=1;
    allAtoms.push_back(atoms[n]);
    for(int r=0;r<ops.size();r++)
    {
      atom1=atoms[n];
      symOp(atoms[n],atom1,0,0,ops[r],ispher);
      found=false;
      for(int m=0;m<allAtoms.size();m++)
      {
	if(allAtoms[m].x==atom1.x && allAtoms[m].y==atom1.y && allAtoms[m].z==atom1.z)
	{
	  found=true;
	}
      }
      if(!found)
      {
	num++;
	allAtoms.push_back(atom1);
      }
    }
    numIdent.push_back(num);
    nMO += num*nMOs[n];
  }

  arma::rowvec character(irreps.size());
  std::vector<arma::mat> mats;
  arma::rowvec groups;
  arma::vec v1,v2,v3;
  arma::mat m1;
  std::vector<std::vector<arma::vec> > vs(irreps.size());

  v3.set_size(nMO);
  //std::vector<std::vector<arma::vec> > Cs(irreps.size());

  int nA,nOrb,nStart;
  num=0;
  nA=0;
  nOrb = 0;
  int numL;
  nA=num;
  for(int o=nA;o<obs.size();o++)
  {
    nA = shell2atom[o];
    nStart=0;
    for(int n=0;n<nA;n++)
    {
      nStart += numIdent[n];
    }
    l = obs[o].contr[0].l;
    if(ispher)
    {
      numL=2*l+1;
    }
    else
    {
      numL=l+1+l*(l+1)/2;
    }
    m1.resize(numIdent[nA],numIdent[nA]); m1.zeros();
    mats.clear();
    v1.resize(numIdent[nA]);
    v2.resize(numIdent[nA]);
    for(int i=0;i<irreps.size();i++)
    {
      mats.push_back(m1);
    }
    for(int nl=0;nl<numL;nl++)
    {
      character.zeros();
      for(int r=0;r<irreps.size();r++)
      {
	mats[r].zeros();
	for(int p=0;p<numIdent[nA];p++)
	{
	  for(int q=0;q<numIdent[nA];q++)
	  {
	    atom1 = allAtoms[nStart+q];
	    mats[r](p,q) = symOp(allAtoms[nStart+p],atom1,l,nl,ops[r],ispher);
	  }
	}
	character(r)=trace(mats[r]);
      }//r
      reduce(character,charTable,groups);
      for(int n=0;n<groups.n_cols;n++)
      {
	if( fabs(groups(n)) > 0.000001)
	{
	  v1.zeros(); v2.zeros();
	  v1(0)=1.0;
	  for(int j=0;j<charTable.n_cols;j++)
	  {
	    v2 += charTable(n,j)*mats[j]*v1;
	  }
	  v3.zeros();
	  for(int j=0;j<v2.n_rows;j++)
	  {
	    v3(nOrb+nl+j*nMOs[nA]) = v2(j);
	  }
	  vs[n].push_back(v3);
	}
      }
    }
    nOrb += numL;
  }

  for(int r=0;r<vs.size();r++)
  {
    m1.set_size(nMO,vs[r].size());
    for(int j=0;j<vs[r].size();j++)
    {
      m1.col(j)=vs[r][j];
    }
    Cs.push_back(m1);
  }
  

  return;
}

void initPG(std::string pg,std::vector<std::string> &irreps,std::vector<std::string> &ops,arma::mat &charTable,arma::Mat<int> &prodTable)
{
  if(pg == "c1")
  {
    irreps.push_back("A");
    ops.push_back("e");
    charTable.set_size(1,1);
    charTable(0,0)=1;
    prodTable.set_size(1,1);
    prodTable(0,0)=0;
  }
  else if(pg == "cs")
  {
    irreps.push_back("Ap"); irreps.push_back("App");
    ops.push_back("e"); ops.push_back("vxy");

    charTable.set_size(2,2); charTable.ones();
    charTable(1,1)=-1;

    prodTable.set_size(2,2);
    prodTable(0,0)=0; prodTable(0,1)=1;
    prodTable(1,0)=1; prodTable(1,1)=0;
  }
  else if(pg == "ci")
  {
    irreps.push_back("Ag"); irreps.push_back("Au");
    ops.push_back("e"); ops.push_back("i");

    charTable.set_size(2,2); charTable.ones();
    charTable(1,1)=-1;

    prodTable.set_size(2,2);
    prodTable(0,0)=0; prodTable(0,1)=1;
    prodTable(1,0)=1; prodTable(1,1)=0;
  }
  else if(pg == "c2")
  {
    irreps.push_back("A"); irreps.push_back("B");
    ops.push_back("e"); ops.push_back("c2z");

    charTable.set_size(2,2); charTable.ones();
    charTable(1,1)=-1;

    prodTable.set_size(2,2);
    prodTable(0,0)=0; prodTable(0,1)=1;
    prodTable(1,0)=1; prodTable(1,1)=0;
  }
  else if(pg == "c2h")
  {
    irreps.push_back("Ag"); irreps.push_back("Bg");
    irreps.push_back("Au"); irreps.push_back("Bu");
    ops.push_back("e"); ops.push_back("c2z");
    ops.push_back("i"); ops.push_back("vxy");

    charTable.set_size(4,4); charTable.ones();
    charTable(1,1)=-1; charTable(1,3)=-1;
    charTable(2,2)=-1; charTable(2,3)=-1;
    charTable(3,1)=-1; charTable(3,2)=-1;

    prodTable.set_size(4,4);
    prodTable(0,0)=0; prodTable(0,1)=1; prodTable(0,2)=2; prodTable(0,3)=3;
    prodTable(1,0)=1; prodTable(1,1)=0; prodTable(1,2)=3; prodTable(1,3)=2;
    prodTable(2,0)=2; prodTable(2,1)=3; prodTable(2,2)=0; prodTable(2,3)=1;
    prodTable(3,0)=3; prodTable(3,1)=2; prodTable(3,2)=1; prodTable(3,3)=0;
  }
  else if(pg == "c2v")
  {
    irreps.push_back("A1"); irreps.push_back("A2");
    irreps.push_back("B1"); irreps.push_back("B2");
    ops.push_back("e"); ops.push_back("c2z");
    ops.push_back("vxz"); ops.push_back("vyz");

    charTable.set_size(4,4); charTable.ones();
    charTable(1,2)=charTable(1,3)=-1;
    charTable(2,1)=charTable(2,3)=-1;
    charTable(3,1)=charTable(3,2)=-1;

    prodTable.set_size(4,4);
    prodTable(0,0)=0; prodTable(0,1)=1; prodTable(0,2)=2; prodTable(0,3)=3;
    prodTable(1,0)=1; prodTable(1,1)=0; prodTable(1,2)=3; prodTable(1,3)=2;
    prodTable(2,0)=2; prodTable(2,1)=3; prodTable(2,2)=0; prodTable(2,3)=1;
    prodTable(3,0)=3; prodTable(3,1)=2; prodTable(3,2)=1; prodTable(3,3)=0;
  }
  else if(pg == "d2h")
  {
    irreps.push_back("Ag"); irreps.push_back("B1g");
    irreps.push_back("B2g"); irreps.push_back("B3g");
    irreps.push_back("Au"); irreps.push_back("B1u");
    irreps.push_back("B2u"); irreps.push_back("B3u");

    ops.push_back("e"); ops.push_back("c2z");
    ops.push_back("c2y"); ops.push_back("c2x");
    ops.push_back("i"); ops.push_back("vxy");
    ops.push_back("vxz"); ops.push_back("vyz");

    charTable.set_size(8,8); charTable.ones();
    charTable(1,2)=-1; charTable(1,3)=-1;
    charTable(1,6)=-1; charTable(1,7)=-1;
    charTable(2,1)=-1; charTable(2,3)=-1;
    charTable(2,5)=-1; charTable(2,7)=-1;
    charTable(3,1)=-1; charTable(3,2)=-1;
    charTable(3,5)=-1; charTable(3,6)=-1;
    charTable(4,4)=-1; charTable(4,5)=-1;
    charTable(4,6)=-1; charTable(4,7)=-1;
    charTable(5,2)=-1; charTable(5,3)=-1;
    charTable(5,4)=-1; charTable(5,5)=-1;
    charTable(6,1)=-1; charTable(6,3)=-1;
    charTable(6,4)=-1; charTable(6,6)=-1;
    charTable(7,1)=-1; charTable(7,2)=-1;
    charTable(7,4)=-1; charTable(7,7)=-1;

    prodTable.set_size(8,8);
    arma::rowvec prod;
    bool same;
    for(int r=0;r<8;r++)
    {
      for(int q=0;q<=r;q++)
      {
	prod=charTable.row(r)%charTable.row(q);
	for(int t=0;t<charTable.n_rows;t++)
	{
	  same=true;
	  for(int i=0;i<prod.n_cols;i++)
	  {
	    if(charTable(t,i)!=prod(i))
	    {
	      same=false;
	    }
	  }
	  if(same)
	  {
	    prodTable(r,q)=t;
	    prodTable(q,r)=t;
	  }
	}
      }
    }

  }

  return;
}

void applySym2(const std::vector<libint2::Atom> &atoms,std::string basis,std::string pointGroup,std::vector<libint2::Atom> &allAtoms,std::vector<int> &nAOs,std::vector<std::vector<std::vector<orbPair> > > &ao2mo,std::vector<std::vector<std::vector<int> > > &bas,arma::Mat<int> &prodTable,std::vector<std::vector<int> > &modes)
{
  std::vector<std::string> ops;
  arma::mat charTable;
  std::vector<std::string> irreps;
  initPG(pointGroup,irreps,ops,charTable,prodTable);

  bas.resize(irreps.size());
  std::vector<int> b(2);

  for(int t=0;t<irreps.size();t++)
  {
    for(int p=0;p<irreps.size();p++)
    {
      for(int q=0;q<irreps.size();q++)
      {
	if(prodTable(p,q)==t)
	{
	  b[0]=p; b[1]=q;
	  bas[t].push_back(b);
        }
      }
    }
  }

  libint2::BasisSet obs(basis,atoms);
  auto shell2atom=obs.shell2atom(atoms);
  int l;
  bool ispher=false;
  for(int i=0;i<basis.size()-1;i++)
  {
    if(basis[i]=='c' && basis[i+1]=='c')
    {
      ispher=true;
    }
  }

  std::vector<int> aosPa;
  for(int n=0;n<obs.size();n++)
  {
    l=obs[n].contr[0].l;
    if(shell2atom[n]+1 > aosPa.size())
    {
      if(ispher)
      {
	aosPa.push_back(2*l+1);
      }
      else
      {
	aosPa.push_back(l+1+l*(l+1)/2);
      }
    }
    else
    {
      if(ispher)
      {
	aosPa[shell2atom[n]] += 2*l+1;
      }
      else
      {
	aosPa[shell2atom[n]] += l+1+l*(l+1)/2;
      }
    }
  }

  libint2::Atom atom1;
 
  int num;
  bool found;
  std::vector<int> numIdent;
//nAOs here
  int nAO=0;
  for(int n=0;n<atoms.size();n++)
  {
    num=1;
    allAtoms.push_back(atoms[n]);
    for(int r=0;r<ops.size();r++)
    {
      atom1=atoms[n];
      symOp(atoms[n],atom1,0,0,ops[r],ispher);
      found=false;
      for(int m=0;m<allAtoms.size();m++)
      {
	if(allAtoms[m].x==atom1.x && allAtoms[m].y==atom1.y && allAtoms[m].z==atom1.z)
	{
	  found=true;
	}
      }
      if(!found)
      {
	num++;
	allAtoms.push_back(atom1);
      }
    }
    numIdent.push_back(num);
    nAO += num*aosPa[n];
  }
 

  std::vector<std::vector<orbPair> > op(irreps.size());
  for(int i=0;i<nAO;i++)
  {
    ao2mo.push_back(op);
  }

  arma::rowvec character(irreps.size());
  std::vector<arma::mat> mats;
  arma::rowvec groups;
  arma::vec v1,v2,v3;
  arma::mat m1;
  std::vector<std::vector<arma::vec> > vs(irreps.size());

  v3.set_size(nAO);
  //std::vector<std::vector<arma::vec> > Cs(irreps.size());

//here here here

  int nA,nOrb,nStart,nAold;
  num=0;
  nA=0;
  nOrb = 0;
  int numL;
  nA=num;
  std::vector<int> numMOs(irreps.size(),0);
  orbPair pair;

  int oStart=0;
  nStart=0;
  int o=0;
  for(int n=0;n<atoms.size();n++)
  {
    nOrb=0;
    while(shell2atom[o]==n)
    {
      l = obs[o].contr[0].l;
      if(ispher)
      {
	numL=2*l+1;
      }
      else
      { 
	numL=l+1+l*(l+1)/2;
      }
      m1.resize(numIdent[n],numIdent[n]); m1.zeros();
      mats.clear();
      v1.resize(numIdent[n]);
      v2.resize(numIdent[n]);
      for(int i=0;i<irreps.size();i++)
      {
	mats.push_back(m1);
      }
      for(int nl=0;nl<numL;nl++)
      {
	character.zeros();
	for(int r=0;r<irreps.size();r++)
	{
	  mats[r].zeros();
	  for(int p=0;p<numIdent[n];p++)
	  {
	    for(int q=0;q<numIdent[n];q++)
	    {
	      atom1 = allAtoms[nStart+q];
	      mats[r](p,q) = symOp(allAtoms[nStart+p],atom1,l,nl,ops[r],ispher);
	    }
	  }
	  character(r)=trace(mats[r]);
	}//r
	reduce(character,charTable,groups);
	for(int r=0;r<groups.n_cols;r++)
	{
	  if( fabs(groups(r)) > 0.000001)
	  {
	    v1.zeros(); v2.zeros();
	    v1(0)=1.0;
	    for(int j=0;j<charTable.n_cols;j++)
	    {
	      v2 += charTable(r,j)*mats[j]*v1;
	    }
//std::cout << o << '\t' << r << std::endl;
//v2.print();
	    v3.zeros();
	    for(int j=0;j<v2.n_rows;j++)
	    {
	      v3(oStart+nOrb+nl+j*aosPa[n]) = v2(j);
	      pair.orb=numMOs[r];
	      pair.weight=v2(j);
	      ao2mo[oStart+nOrb+nl+j*aosPa[n]][r].push_back(pair);
	    }
	    vs[r].push_back(v3);
	    numMOs[r] += 1;
	  }
        }
      }

//std::cout << o << '\t' << numL << std::endl;
      nOrb += numL;
      o++;
    }

    nStart += numIdent[n];
    oStart += aosPa[n]*numIdent[n];
  }
  for(int r=0;r<vs.size();r++)
  {
    nAOs.push_back(vs[r].size());
  }
/*  return;

  for(int o=0;o<obs.size();o++)
  {
    nAold=nA;
    nA = shell2atom[o];
    if(nAold != nA)
    {
      nStart=0;
      num=0;
      for(int n=0;n<nA;n++)
      {
	nStart += numIdent[n];
	num += aosPa[n]*numIdent[n];
      }
      nOrb=0;
    }
    l = obs[o].contr[0].l;
    if(ispher)
    {
      numL=2*l+1;
    }
    else
    {
      numL=l+1+l*(l+1)/2;
    }
    m1.resize(numIdent[nA],numIdent[nA]); m1.zeros();
    mats.clear();
    v1.resize(numIdent[nA]);
    v2.resize(numIdent[nA]);
    for(int i=0;i<irreps.size();i++)
    {
      mats.push_back(m1);
    }
    for(int nl=0;nl<numL;nl++)
    {
      character.zeros();
      for(int r=0;r<irreps.size();r++)
      {
	mats[r].zeros();
	for(int p=0;p<numIdent[nA];p++)
	{
	  for(int q=0;q<numIdent[nA];q++)
	  {
	    atom1 = allAtoms[nStart+q];
	    mats[r](p,q) = symOp(allAtoms[nStart+p],atom1,l,nl,ops[r],ispher);
	  }
	}
	character(r)=trace(mats[r]);
      }//r
      reduce(character,charTable,groups);
      for(int n=0;n<groups.n_cols;n++)
      {
	if( fabs(groups(n)) > 0.000001)
	{
	  v1.zeros(); v2.zeros();
	  v1(0)=1.0;
	  for(int j=0;j<charTable.n_cols;j++)
	  {
	    v2 += charTable(n,j)*mats[j]*v1;
	  }
	  v3.zeros();
	  for(int j=0;j<v2.n_rows;j++)
	  {
	    v3(nOrb+nl+j*aosPa[nA]) = v2(j);
	    pair.orb=numMOs[n];
	    pair.weight=v2(j);
	    ao2mo[num+nOrb+nl+j*aosPa[nA]][n].push_back(pair);
	  }
	  vs[n].push_back(v3);
	  numMOs[n] += 1;
	}
      }
    }
    std::cout << o << '\t' << nA << '\t' << nOrb << std::endl;
    nOrb += numL;
  }

  for(int r=0;r<vs.size();r++)
  {
    nAOs.push_back(vs[r].size());
  }
*/
  std::vector<std::vector<bool> > frozen(atoms.size(),std::vector<bool>(3,false));
  for(int a=0;a<atoms.size();a++)
  {
    if(pointGroup=="c2v" || pointGroup=="d2h" || pointGroup=="d2")
    {
      if(atoms[a].x == 0.0)
      {
	frozen[a][0] = true;
      }
      if(atoms[a].y == 0.0)
      {
	frozen[a][1] = true;
      }
    }
    if(pointGroup=="cs" || pointGroup=="c2h" || pointGroup=="d2h" || pointGroup=="d2")
    {
      if(atoms[a].z == 0.0)
      {
	frozen[a][2] = true;
      }
    }
    if(pointGroup=="c2" || pointGroup=="c2h")
    {
      if(atoms[a].x==0.0 && atoms[a].y==0.0)
      {
	frozen[a][0] = true;
	frozen[a][1] = true;
      }
    }
    if(pointGroup=="ci")
    {
      if(atoms[a].x==0.0 && atoms[a].y==0.0 && atoms[a].z==0.0)
      {
	frozen[a][0] = true;
	frozen[a][1] = true;
	frozen[a][2] = true;
      }
    }
  }

  std::vector<double> rCM(3,0.0);
  double M,m;
  M=0.0;
  for(int a=0;a<allAtoms.size();a++)
  {
    m = allAtoms[a].atomic_number;
    rCM[0] += m*allAtoms[a].x;
    rCM[1] += m*allAtoms[a].y;
    rCM[2] += m*allAtoms[a].z;
    M += m;
  }
  for(int i=0;i<3;i++)
  {
    rCM[i] /= M;
  }
  int centroid=0;
  double Rold,Rnew;
  Rold=100000000000;
  for(int a=0;a<atoms.size();a++)
  {
    Rnew = (atoms[a].x-rCM[0])*(atoms[a].x-rCM[0]);
    Rnew += (atoms[a].y-rCM[1])*(atoms[a].y-rCM[1]);
    Rnew += (atoms[a].z-rCM[2])*(atoms[a].z-rCM[2]);
    if(Rnew < Rold)
    {
      centroid=a;
      Rold = Rnew;
    }
  }

  if(pointGroup=="c2" || pointGroup=="c2v")
  {
    frozen[centroid][2]=true;
  }
  if(pointGroup=="c1")
  {
    frozen[centroid][0]=true;
    frozen[centroid][1]=true;
    frozen[centroid][2]=true;
  }
  if(pointGroup=="cs")
  {
    frozen[centroid][0]=true;
    frozen[centroid][1]=true;
  }

  std::vector<int> vec(2);
  for(int a=0;a<atoms.size();a++)
  {
    vec[0]=a;
    for(int i=0;i<3;i++)
    {
      if(!frozen[a][i])
      {
	vec[1]=i;
	modes.push_back(vec);
      }
    }
  }

  return;
}

void genAtoms(std::string basis,std::string pointGroup,const std::vector<libint2::Atom> &atoms,std::vector<libint2::Atom> &atomsFull)
{
  std::vector<std::string> ops;
  arma::mat charTable;
  std::vector<std::string> irreps;
  arma::Mat<int> prodTable;
  initPG(pointGroup,irreps,ops,charTable,prodTable);

  libint2::Atom atom1;
  bool found;
  atomsFull.clear();
  for(int n=0;n<atoms.size();n++)
  {
    atomsFull.push_back(atoms[n]);
    for(int r=0;r<ops.size();r++)
    {
      atom1=atoms[n];
      symOp(atoms[n],atom1,0,0,ops[r],false);
      found=false;
      for(int m=0;m<atomsFull.size();m++)
      {
	if(atomsFull[m].x==atom1.x && atomsFull[m].y==atom1.y && atomsFull[m].z==atom1.z)
	{
	  found=true;
	}
      }
      if(!found)
      {
	atomsFull.push_back(atom1);
      }
    }
  }

  return;
}

void atomStartEnd(std::string pointGroup,const std::vector<libint2::Atom> &atoms,int numA,int &start,int &end)
{
  std::vector<std::string> ops;
  arma::mat charTable;
  std::vector<std::string> irreps;
  arma::Mat<int> prodTable;
  initPG(pointGroup,irreps,ops,charTable,prodTable);

  std::vector<libint2::Atom> atomsFull;  
  libint2::Atom atom1;
  bool found;
  int n=0;
  while(n != numA)
  //for(int n=0;n<atoms.size();n++)
  {
    atomsFull.push_back(atoms[n]);
    for(int r=0;r<ops.size();r++)
    {
      atom1=atoms[n];
      symOp(atoms[n],atom1,0,0,ops[r],false);
      found=false;
      for(int m=0;m<atomsFull.size();m++)
      {
	if(atomsFull[m].x==atom1.x && atomsFull[m].y==atom1.y && atomsFull[m].z==atom1.z)
	{
	  found=true;
	}
      }
      if(!found)
      {
	atomsFull.push_back(atom1);
      }
    }
    n++;
  }

  start = atomsFull.size();
  atomsFull.push_back(atoms[n]);
  for(int r=0;r<ops.size();r++)
  {
    atom1=atoms[n];
    symOp(atoms[n],atom1,0,0,ops[r],false);
    found=false;
    for(int m=0;m<atomsFull.size();m++)
    {
      if(atomsFull[m].x==atom1.x && atomsFull[m].y==atom1.y && atomsFull[m].z==atom1.z)
      {
	found=true;
      }
    }
    if(!found)
    {
      atomsFull.push_back(atom1);
    }
  }
  end = atomsFull.size()-1;

  return;
}
