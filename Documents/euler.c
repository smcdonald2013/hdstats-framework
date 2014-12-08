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

//Free the memory associated with integrator 
void integrator_free(Integrator *integrator){
free(integrator);
}

//Advance one timestep 
int integrator_step(Integrator *integrator, double t, double *x){
  FuncPtr modelCalc = integrator->rhs;
  const int n = integrator->n;
  const double dt = integrator->dt;
  double fx[n];

  modelCalc(n,t,x,fx);
  
  //Performing euler update for each variable
  int i;
  for(i=0;i<n;i++){
    x[i]=x[i]+dt*fx[i];
  }
  return 0;
}


