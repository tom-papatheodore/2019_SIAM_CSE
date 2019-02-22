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
#endif /*_OPENACC*/

#include <mpi.h>

//#define max(x, y) (((x) > (y)) ? (x) : (y))
//#define min(x, y) (((x) < (y)) ? (x) : (y))

#define NY 4096
#define NX 4096

double A[NY][NX];
double Anew[NY][NX];
double rhs[NY][NX];

double A_ref[NY][NX];
double Anew_ref[NY][NX];

#include "poisson2d_serial.h"
void poisson2d_serial(int , double);

int min( int a, int b)
{
    return a < b ? a : b;
}

int max( int a, int b)
{
    return a > b ? a : b;
}

int main(int argc, char** argv)
{
    int iter_max = 1000;
    const double tol = 1.0e-5;

	int rank = 0;
	int size = 1;

	//Initialize MPI and determine rank and size
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

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

#if _OPENACC
    acc_device_t device_type = acc_get_device_type();
    if ( acc_device_nvidia == device_type )
    {
        int ngpus=acc_get_num_devices(acc_device_nvidia);
    
        int devicenum=rank%ngpus;
        acc_set_device_num(devicenum,acc_device_nvidia);
    }
    // Call acc_init after acc_set_device_num to avoid multiple contexts on device 0 in multi GPU systems
    acc_init(device_type);
#endif /*_OPENACC*/

	#pragma acc enter data create(A,A_ref,Anew,rhs)
 
    int ix_start = 1;
    int ix_end   = (NX - 1);

    // Ensure correctness if NY%size != 0
    int chunk_size = ceil( (1.0*NY)/size );

    int iy_start = rank * chunk_size;
    int iy_end   = iy_start + chunk_size;

    // Do not process boundaries
    iy_start = max( iy_start, 1 );
    iy_end = min( iy_end, NY - 1 );

	// set A and A_ref 
    //OpenACC Warm-up
    #pragma acc kernels
    for(int iy = 0; iy < NY; iy++)
    {
        for(int ix = 0; ix < NX; ix++)
        {
            A_ref[iy][ix] = 0.0;
            A[iy][ix]    = 0.0;
        }
    }

    if ( rank == 0) printf("Jacobi relaxation Calculation: %d x %d mesh\n", NY, NX);
    if ( rank == 0) printf("Calculate reference solution and time serial execution.\n");
	fflush(stdout);

	if ( rank == 0){
	// Serial Execution
	printf("Serial Execution...\n");
	fflush(stdout);
	gettimeofday(&start_time, NULL);
    poisson2d_serial(iter_max, tol);
    gettimeofday(&stop_time, NULL);
    timersub(&stop_time, &start_time, &elapsed_time_serial);
	}

    //MPI Warm-up to establish CUDA IPC connections
    for (int i=0; i<2; ++i)
    {
        int top    = (rank == 0) ? (size-1) : rank-1;
        int bottom = (rank == (size-1)) ? 0 : rank+1;
        #pragma acc host_data use_device( A )
        {
            //1. Sent row iy_start (first modified row) to top receive lower boundary (iy_end) from bottom
            MPI_Sendrecv( &A[iy_start][ix_start], (ix_end-ix_start), MPI_DOUBLE, top   , 0, &A[iy_end][ix_start], (ix_end-ix_start), MPI_DOUBLE, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            //2. Sent row (iy_end-1) (last modified row) to bottom receive upper boundary (iy_start-1) from top
            MPI_Sendrecv( &A[(iy_end-1)][ix_start], (ix_end-ix_start), MPI_DOUBLE, bottom, 0, &A[(iy_start-1)][ix_start], (ix_end-ix_start), MPI_DOUBLE, top   , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
    }

	// Parallel Execution
    //Wait for all processes to ensure correct timing of the parallel version
    MPI_Barrier( MPI_COMM_WORLD );
	if ( rank == 0) {
		printf("Parallel execution.\n");
		fflush(stdout);
	}
	gettimeofday(&start_time, NULL);

    int iter  = 0;
    double error = 1.0;
   
	#pragma acc update device(A[(iy_start-1):(iy_end-iy_start)+2][0:NX],rhs[iy_start:(iy_end-iy_start)][0:NX]) 
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

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
      
		double global_error = 0.0;
		MPI_Allreduce(&error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		error = global_error;
 
		#pragma acc kernels
        for (int iy = iy_start; iy < iy_end; iy++)
        {
            for( int ix = ix_start; ix < ix_end; ix++ )
            {
                A[iy][ix] = Anew[iy][ix];
            }
        }
       
        //Periodic boundary conditions
        int top    = (rank == 0) ? (size-1) : rank-1;
        int bottom = (rank == (size-1)) ? 0 : rank+1;
        #pragma acc host_data use_device( A )
        {
            //1. Sent row iy_start (first modified row) to top receive lower boundary (iy_end) from bottom
            MPI_Sendrecv( &A[iy_start][ix_start], (ix_end-ix_start), MPI_DOUBLE, top   , 0, &A[iy_end][ix_start], (ix_end-ix_start), MPI_DOUBLE, bottom, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

            //2. Sent row (iy_end-1) (last modified row) to bottom receive upper boundary (iy_start-1) from top
            MPI_Sendrecv( &A[(iy_end-1)][ix_start], (ix_end-ix_start), MPI_DOUBLE, bottom, 0, &A[(iy_start-1)][ix_start], (ix_end-ix_start), MPI_DOUBLE, top   , 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
        }
 
		#pragma acc kernels
        for (int iy = iy_start; iy < iy_end; iy++)
        {
                A[iy][0]      = A[iy][(NX-2)];
                A[iy][(NX-1)] = A[iy][1];
        }
       
        if((rank == 0) && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, error);
 
        iter++;
    }

	#pragma acc update self(A[(iy_start-1):(iy_end-iy_start)+2][0:NX])

	MPI_Barrier( MPI_COMM_WORLD );

	gettimeofday(&stop_time, NULL);
	timersub(&stop_time, &start_time, &elapsed_time_parallel);

	double runtime_serial;
	if (rank == 0) {runtime_serial   = elapsed_time_serial.tv_sec+elapsed_time_serial.tv_usec/1000000.0;}
    double runtime_parallel = elapsed_time_parallel.tv_sec+elapsed_time_parallel.tv_usec/1000000.0;
	
	if(rank == 0) printf("Elapsed Time (s) - Serial: %8.4f, Parallel: %8.4f, Speedup: %8.4f\n", runtime_serial, runtime_parallel, runtime_serial/runtime_parallel);

	#pragma acc exit data delete(A,A_ref,Anew,rhs)

	MPI_Finalize();

    return 0;
}
