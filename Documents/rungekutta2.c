#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "integrator.h"

//Defining FuncPtr
//For now, it is just a dummy test function
int myFun(int n, double t, const double *x, double *fx){
  double data;
  data = *x;
  *fx = data+2;  
  return 0;
}

//Defining a new integrator object
struct integrator_t{
  int n;
  double dt;
  FuncPtr rhs;
};

//Returning a new integrator object
Integrator *integrator_new(int n, double dt, FuncPtr rhs){
  Integrator *intNew;
  intNew = malloc(sizeof(Integrator));
  intNew->n = n;
  intNew->dt = dt;
  intNew->rhs = rhs;
  return intNew;
 }

void integrator_free(Integrator *integrator){
free(integrator);
}

int integrator_step(Integrator *integrator, double t, double *x){
  FuncPtr thisCalc = integrator->rhs;
  double dt = integrator->dt;
  int n = integrator->n;
  double fx[n];
  int i;
  for(i=0;i<n;i++){
    fx[i]=x[i];
  }
  thisCalc(n,t,x,fx);
  double k1[n];
  double newX[n];
  for(i=0;i<n;i++){
    k1[i] = fx[i];
    newX[i] = x[i]+k1[i]*dt/2; 
  }
  thisCalc(n,t+dt/2,newX,fx);
  double k2[n];
  for(i=0;i<n;i++){
    k2[i] = fx[i];
    newX[i] = x[i]+k2[i]*dt/2; 
  }
  thisCalc(n,t+dt/2,newX,fx);
  double k3[n];
  for(i=0;i<n;i++){
    k3[i] = fx[i];
    newX[i] = x[i]+dt*k3[i]; 
  }
  thisCalc(n,t+dt/2,newX,fx);
  double k4[n];
  for(i=0;i<n;i++){
    k4[i] = fx[i]; 
    x[i] = x[i]+dt*k1[i]/6+dt*(k2[i]+k3[i])/3+dt*k4[i]/6;
  }
  return 0;
}
