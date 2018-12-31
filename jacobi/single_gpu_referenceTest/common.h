#ifndef COMMON_H
#define COMMON_H

void poisson2d_serial(int max_iter, double tol);

#define NY 2048
#define NX 2048

double A[NX][NY];
double Anew[NX][NY];
double Aref[NX][NY];
double rhs[NX][NY];

#endif // COMMON_H
