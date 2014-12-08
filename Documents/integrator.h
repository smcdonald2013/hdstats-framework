#ifndef INTEGRATOR_H
#define INTEGRATOR_H

typedef int (*FuncPtr)(int n, double t, const double *x, double *fx);

typedef struct integrator_t Integrator;

Integrator *integrator_new(int n, double dt, FuncPtr rhs);

void integrator_free(Integrator *integrator);

int integrator_step(Integrator *integrator, double t, double *x);

#endif
