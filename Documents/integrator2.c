#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "integrator.h"

//Defining a new integrator object
struct integrator_t{
  int n;
  double dt;
  FuncPtr rhs;
};

//Returning a new integrator object
Integrator *integrator_new(int n, double dt, FuncPtr rhs){
  Integrator *intNew = malloc(sizeof(Integrator));
  assert(intNew != NULL);
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
  int n = integrator->n;
  double fx[n];
  int i;
  for(i=0;i<n;i++){
    fx[i]=x[i];
  }
  thisCalc(integrator->n,t,x,fx);
  //printf("%15.8f",t);
  //printf("%15.8f",x[0]);
  //printf("%15.8f",x[1]);
  x[0]=x[0]+integrator->dt*fx[0];
  x[1]=x[1]+integrator->dt*fx[1];
  return 0;
}

