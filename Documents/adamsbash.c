#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "integrator.h"

//Defining a new integrator object
struct integrator_t{
  int n;
  double dt;
  FuncPtr rhs;
  double *fxOld;
};

//Returning a new integrator object
Integrator *integrator_new(int n, double dt, FuncPtr rhs){
  Integrator *intNew = malloc(sizeof(Integrator));
  intNew->n = n;
  intNew->dt = dt;
  intNew->rhs = rhs;
  intNew->fxOld = malloc(n*sizeof(double));
  return intNew;
 }

void integrator_free(Integrator *integrator){
  free(integrator->fxOld);
  free(integrator);
}

int integrator_step(Integrator *integrator, double t, double *x){
  FuncPtr thisCalc = integrator->rhs;
  int n = integrator->n;
  double dt = integrator->dt;
  double fx[n];
  int i;
  for(i=0;i<n;i++){
    fx[i]=x[i];
  }
  if(t==0){
  thisCalc(integrator->n,t,x,fx);
  x[0]=x[0]+integrator->dt*fx[0];
  integrator->fxOld[0]=fx[0];
  x[1]=x[1]+integrator->dt*fx[1];
  integrator->fxOld[1]=fx[1];
  }
  else{
    thisCalc(integrator->n,t,x,fx);
    printf("%15.8f\n",integrator->fxOld[1]);
    for(i=0;i<n;i++){
      x[i]=x[i]+1.5*dt*fx[i]-.5*dt*integrator->fxOld[i];
      integrator->fxOld[i] = fx[i];
    }
    printf("%15.8f\n",integrator->fxOld[1]);
  }
  return 0;
}
