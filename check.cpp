#include "stdafx.h"
#include <iomanip>
#include <iostream>
#include "math.h"
#include "time.h"
#include "optimization.h"

using namespace alglib;

void function1_grad(const real_1d_array &x,double &func,real_1d_array &grad,void *ptr)
{
  func = 100*pow(x[0]+3,4) + pow(x[1]-3,4);
  grad[0] = 400*pow(x[0]+3,3);
  grad[1] = 4*pow(x[1]-3,3);

  std::string *strPtr=static_cast<std::string*>(ptr);
  std::cout << *strPtr << std::endl;
}

void printIt(const real_1d_array &x,double func,void *ptr)
{
  std::cout << '\t' << func << std::endl;
}

int main(int argc, char *argv[])
{

  real_1d_array x = "[0,0]";
  double epsg = 0.000000001;
  double epsf = 0;
  double epsx = 0;
  ae_int_t maxits = 0;
  minlbfgsstate state;
  minlbfgsreport rep;

  std::string str="hello world!";

  //number of Hessian updates, array of variables, state variable used for
  //error messaging
  minlbfgscreate(1,x,state);
  //state variable
  //EpsG is threshold for scaled gradient vector (want this one)
 	//requires MinLBFGSSetScale
  //EpsF is threshold for energy differences (don't use)
  //EpsX is threshold for step size (again, don't use)
  //MaxIts, duh
  minlbfgssetcond(state,epsg,epsf,epsx,maxits);
  minlbfgssetxrep(state,true);
  //pass it a function (see above) to evaluate function and gradient
  minlbfgsoptimize(state,function1_grad,printIt,&str);
  minlbfgsresults(state,x,rep);

  printf("%d\n",int(rep.terminationtype));
//  printf("%s\n",x.tostring(2).c_str());
  std::cout << std::setprecision(10);
  for(int i=0;i<2;i++)
  {
    std::cout << x[i] << std::endl;
  }


  return EXIT_SUCCESS;
}

