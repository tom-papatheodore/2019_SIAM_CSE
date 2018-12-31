/* Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <sys/time.h>

#define NY 2048
#define NX 2048

double A[NY][NX];
double Anew[NY][NX];
double rhs[NY][NX];

double A_ref[NY][NX];
double Anew_ref[NY][NX];

#include "poisson2d_serial.h"
void poisson2d_serial(int , double);

int main(int argc, char** argv)
{
    int iter_max = 1000;
    const double tol = 1.0e-5;

	struct timeval start_time, stop_time, elapsed_time_serial, elapsed_time_parallel;

    // set rhs
    for (int iy = 1; iy < NY-1; iy++)
    {
        for( int ix = 1; ix < NX-1; ix++ )
        {
            const double x = -1.0 + (2.0*ix/(NX-1));
            const double y = -1.0 + (2.0*iy/(NY-1));
            rhs[iy][ix] = exp(-10.0*(x*x + y*y));
        }
    }
  
	// set A and A_ref 
    for(int iy = 0; iy < NY; iy++)
    {
        for(int ix = 0; ix < NX; ix++)
        {
            A_ref[iy][ix] = 0.0;
            A[iy][ix]    = 0.0;
        }
    }

	printf("Jacobi relaxation Calculation: %d x %d mesh\n", NY, NX);

	// Serial Execution
	printf("Serial Execution...\n");
	gettimeofday(&start_time, NULL);
    poisson2d_serial(iter_max, tol);
    gettimeofday(&stop_time, NULL);
    timersub(&stop_time, &start_time, &elapsed_time_serial);

	// Parallel Execution
	printf("Parallel Execution...\n"); 
	gettimeofday(&start_time, NULL);

    int iter  = 0;
    double error = 1.0;
   
	#pragma acc data copy(A[0:NY][0:NX]) copyin(rhs[0:NY][0:NX]) create(Anew[0:NY][0:NX])
	{
 
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

		#pragma acc kernels
        for (int iy = 1; iy < NY-1; iy++)
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                Anew[iy][ix] = -0.25 * (rhs[iy][ix] - ( A[iy][ix+1] + A[iy][ix-1]
                                                       + A[iy-1][ix] + A[iy+1][ix] ));
                error = fmax( error, fabs(Anew[iy][ix]-A[iy][ix]));
            }
        }
        
		#pragma acc kernels
        for (int iy = 1; iy < NY-1; iy++)
        {
            for( int ix = 1; ix < NX-1; ix++ )
            {
                A[iy][ix] = Anew[iy][ix];
            }
        }
        
        //Periodic boundary conditions
		#pragma acc kernels
        for( int ix = 1; ix < NX-1; ix++ )
        {
                A[0][ix]      = A[(NY-2)][ix];
                A[(NY-1)][ix] = A[1][ix];
        }
		#pragma acc kernels
        for (int iy = 1; iy < NY-1; iy++)
        {
                A[iy][0]      = A[iy][(NX-2)];
                A[iy][(NX-1)] = A[iy][1];
        }
        
        if((iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }
	} /* #pragma acc data */

	gettimeofday(&stop_time, NULL);
	timersub(&stop_time, &start_time, &elapsed_time_parallel);

	double runtime_serial   = elapsed_time_serial.tv_sec+elapsed_time_serial.tv_usec/1000000.0;
    double runtime_parallel = elapsed_time_parallel.tv_sec+elapsed_time_parallel.tv_usec/1000000.0;
    printf("Elapsed Time (s) - Serial: %8.4f, Parallel: %8.4f, Speedup: %8.4f\n", runtime_serial, runtime_parallel, runtime_serial/runtime_parallel);

    return 0;
}
