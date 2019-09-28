#include "D2.h"

int indexD(int i,int j,int r)
{
  return (i*r - i*(i+3)/2+j-1);
}

//index when i must be greater than j
int indexR(int i,int j)
{
  return i*(i-1)/2+j;
}

int indexD1(int i,int r)
{
  return (i*r - i*(i+3)/2-1);
}

int index(int i,int j,int r)
{
  return (i*r+j);
}

int nChooseR(int n,int r)
{
  if(n < 0 || r < 0 || (n-r<0) )
  {
    std::cerr << "invalid nChooseR" << std::endl;
  }
  int fac=1;
  for(int i=n;i>r;i--)
  {
    fac *= i;
  }
  for(int i=n-r;i>1;i--)
  {
    fac /= i;
  }

  return fac;
}

void setTime(std::chrono::high_resolution_clock::time_point &t1,std::chrono::high_resolution_clock::time_point &t2,std::chrono::duration<double> &time_span)
{
  t2 = std::chrono::high_resolution_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2-t1);

  return;
}
