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

#ifdef _OPENACC
#include <openacc.h>
#endif /* _OPENACC */

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#define max(x, y) (((x) > (y)) ? (x) : (y))
#define min(x, y) (((x) < (y)) ? (x) : (y))

#define NY 2048
#define NX 2048

double A[NX][NY];
double Anew[NX][NY];
double rhs[NX][NY];

int main(int argc, char** argv)
{
    int iter_max = 1000;
    const double tol = 1.0e-5;

	struct timeval start_time, stop_time, elapsed_time;
//    gettimeofday(&start_time, NULL);

    memset(A, 0, NY * NX * sizeof(double));
  
	int num_threads = 1;
	int thread_num  = 0;

	double global_error = 0.0;

	#pragma omp parallel default(shared) firstprivate(num_threads, thread_num)
	{

#ifdef _OPENMP
	num_threads = omp_get_num_threads();
	thread_num  = omp_get_thread_num();	
#endif /* _OPENMP */


	#pragma omp master
	{ 
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
	}    

#ifdef _OPENACC
	int num_devices = acc_get_num_devices(acc_device_nvidia);	
	int device_num  = thread_num % num_devices;
	acc_set_device_num(device_num, acc_device_nvidia);
#endif /* _OPENACC */

	printf("[%d]: GPU %d of %d\n", thread_num, device_num, num_devices);

	#pragma omp master
	{
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", NY, NX);
	gettimeofday(&start_time, NULL);
	}

    int iter  = 0;
    double error = 1.0;
   
	int ix_start = 1;
	int ix_end   = NX;

	/* Use ceil function in case num_threads does not divide evenly into NY */
	int chunk_size = ceil((1.0*NY)/num_threads);

	/* ---------------------------------------------------------------------- 
		For each thread, these values are set so the loops below can iterate
		from iy_start to iy_end-1, which include only the inner region of the 
		domain that need to be calculated.

		They are also used to set the ranges of data that each thread sends
		to its GPU (including the halo region).
	 ----------------------------------------------------------------------*/
	int iy_start = thread_num * chunk_size;
	int iy_end   = iy_start + chunk_size;

	/* Only process inner region - not boundaries */
	iy_start = max(iy_start, 1);
	iy_end   = min(iy_end, NY-1);

	#pragma acc data copy(A[(iy_start-1):(iy_end-iy_start)+2][0:NX]) copyin(rhs[iy_start:(iy_end-iy_start)][0:NX]) create(Anew[iy_start:(iy_end-iy_start)][0:NX])
	{ 
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;
		#pragma omp single
		{
		global_error = 0.0;
		}
		#pragma omp barrier

		#pragma acc kernels
        for (int iy = iy_start; iy < iy_end; iy++)
        {
            for( int ix = ix_start; ix < ix_end; ix++ )
            {
                Anew[iy][ix] = -0.25 * (rhs[iy][ix] - ( A[iy][ix+1] + A[iy][ix-1]
                                                       + A[iy-1][ix] + A[iy+1][ix] ));
                error = fmax( error, fabs(Anew[iy][ix]-A[iy][ix]));
            }
        }
      
//		printf("[1]: error = %f, global_error = %f\n", error, global_error);
 
		#pragma omp critical
		{
		global_error = max(global_error, error);
//		printf("Global Error: %f\n", global_error);
		}

		#pragma omp barrier
		error = global_error;
 
//		printf("[2]: error = %f, global_error = %f\n", error, global_error);

		#pragma acc kernels
        for (int iy = iy_start; iy < iy_end; iy++)
        {
            for( int ix = ix_start; ix < ix_end; ix++ )
            {
                A[iy][ix] = Anew[iy][ix];
            }
        }

        // Begin periodic boundary conditions update
		#pragma acc update self(A[iy_start:1][0:NX], A[(iy_end-1):1][0:NX])

		#pragma omp barrier
		if(0 == (iy_start-1))
		{
        	for( int ix = 1; ix < NX-1; ix++ )
        	{
				A[0][ix]      = A[(NY-2)][ix];
			}
		}

		if((NY-1) == (iy_end))
		{
			for( int ix = 1; ix < NX-1; ix++ )
			{
				A[(NY-1)][ix] = A[1][ix];
			}
		}

		#pragma acc update device(A[(iy_start-1):1][0:NX], A[iy_end:1][0:NX])

		#pragma acc kernels
        for (int iy = iy_start; iy < iy_end; iy++)
        {
                A[iy][0]      = A[iy][(NX-2)];
                A[iy][(NX-1)] = A[iy][1];
        }
       
		#pragma omp master
		{ 
        if((iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
		}        

        iter++;
    }
	} /* #pragma acc data */

	#pragma omp master
	{
	gettimeofday(&stop_time, NULL);
	timersub(&stop_time, &start_time, &elapsed_time);

	printf("%dx%d: 1 CPU: %8.4f s\n", NY, NX, elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
	}

	} /* #pragma omp parallel */

    return 0;
}
