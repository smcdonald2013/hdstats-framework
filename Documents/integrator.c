#include <stdio.h>
#include "integrator.h"

//Defining FuncPtr
//For now, it is just a dummy test function
int myFun(int n, double t, const double *x, double *fx){
  return 0;
}

FuncPtr thisPt  = &myFun;

//Defining a new integrator object
struct integrator_t{
  int n;
  double dt;
  FuncPtr rhs;
};

//Returning a new integrator object
Integrator integratorThis(int n, double dt, FuncPtr rhs){
    Integrator intNew;
    intNew.n = n;
    intNew.dt = dt;
    intNew.rhs = rhs;
    return intNew;
  }

Integrator *integrator_new(int n, double dt, FuncPtr rhs){
  return 0;
}

// = &integratorThis;
