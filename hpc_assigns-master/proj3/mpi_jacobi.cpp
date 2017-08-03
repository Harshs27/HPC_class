/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
/*
 *Solutions implemented by Patrick Lavin and Harsh Shrivastava
 * */


#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"
#include <mpi.h>

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>


void my_get_grid_comm(MPI_Comm* grid_comm)
{
    // get comm size and rank
    int rank, p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int q = (int)sqrt(p);
//    ASSERT_EQ(q*q, p) << "Number of processors must be a perfect square.";

    // split into grid communicator
    int dims[2] = {q, q};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, grid_comm);
}



void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int q = sqrt(comm_size);
    int floor_nq = n/q, ceil_nq = n/q + (n % q != 0);
    // distribute the input_vector among the processors in the first column
    // initially everything is in rank 00
    int rank00, myrank;
    int coords[2] = {0, 0};
    // getting the processor rank corresponding to the cart rank (0, 0)
    MPI_Cart_rank(comm, coords, &rank00);
    MPI_Comm_rank(comm, &myrank); // actual physical rank of processor

    MPI_Comm commcol;// subcomm consisting of columns
    int remain_dims[2] = {1, 0};// select the columns

    MPI_Cart_sub(comm, remain_dims, &commcol);

    // for every processor, get its coordinates in comm world
    MPI_Cart_coords(comm, myrank, 2, coords);
    int ndims[1];
    MPI_Cartdim_get(commcol, ndims);
    int *dims = (int *)malloc(ndims[0]*sizeof(int));
    dims[0] = q;
    int* coords_col_check = (int *)malloc(q*sizeof(int));
    int periods[1] = {0};
    if(coords[1]==0){// in the 1st column
        MPI_Cart_get(commcol, 1, dims, periods, coords_col_check);
        int *sendcounts = (int *)malloc(q*sizeof(int));// commcol size (# of processors)
        int *displacement = (int *)malloc(q*sizeof(int));// commcol size (# of processors)
        int recvcount;
        int commcol_rank, commcol_size;
        MPI_Comm_rank(commcol, &commcol_rank);
        MPI_Comm_size(commcol, &commcol_size);
        if(commcol_rank<n%q)
            recvcount = ceil_nq;
        else
            recvcount = floor_nq;
        // find the rank00 processor in comm world in the corresponding subcomm world. (r_comm-column)/q will give the row rank in column
        int root = (rank00-coords[1])/q;
        if(commcol_rank == root){// distribute data from this processor in its commcol
            displacement[0] = 0;
            if(n>=q){
                for(int i=0; i<q; i++){
                    if(i<n%q)
                        sendcounts[i] = ceil_nq;
                    else
                        sendcounts[i] = floor_nq;
                    if(i<q-1)
                        displacement[i+1] = displacement[i]+sendcounts[i];
                }
            }
            else{// case where number of elements is less than the sqrt(processors)
                for(int i=0; i<q; i++){
                    if(i<n)
                        sendcounts[i] = 1;
                    else
                        sendcounts[i] = 0;
                    displacement[i] = i;
                }
            }
            
        }
        MPI_Scatterv(input_vector, sendcounts, displacement, MPI_DOUBLE, *local_vector, recvcount, MPI_DOUBLE, root, commcol);
        free(sendcounts);
        free(displacement); 
    }// 1st column ends
    MPI_Comm_free(&commcol); 
    free(dims);
    free(coords_col_check);
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int q = sqrt(comm_size);
    int floor_nq = n/q, ceil_nq = n/q + (n % q != 0);
    int rank00, myrank;
    int coords[2] = {0, 0};
    // getting the processor rank corresponding to the cart rank (0, 0)
    MPI_Cart_rank(comm, coords, &rank00);
    MPI_Comm_rank(comm, &myrank); // actual physical rank of processor

    MPI_Comm commcol;// subcomm consisting of columns
    int remain_dims[2] = {1, 0};// select the columns

    MPI_Cart_sub(comm, remain_dims, &commcol);

    // for every processor, get its cartesian coordinates in comm world
    MPI_Cart_coords(comm, myrank, 2, coords);
    int root = (rank00-coords[1])/q;
    if(coords[1]==0){// 1st column
        int commcol_rank, commcol_size;
        MPI_Comm_rank(commcol, &commcol_rank);
        MPI_Comm_size(commcol, &commcol_size);
        int sendcount;
        int *recvcounts = (int *)malloc(q*sizeof(int));// commcol size (# of processors)
        int *displacement = (int *)malloc(q*sizeof(int));// commcol size (# of processors)

        if(commcol_rank<n%q)
            sendcount = ceil_nq;
        else
            sendcount = floor_nq;
        
        if(commcol_rank==root){
            displacement[0] = 0;
            if(n>=q){
                for(int i=0; i<q; i++){
                    if(i<n%q)
                        recvcounts[i] = ceil_nq;
                    else
                        recvcounts[i] = floor_nq;
                    if(i<q-1)
                        displacement[i+1] = displacement[i]+recvcounts[i];
                }
            }
            else{// case where number of elements is less than the sqrt(processors)
                for(int i=0; i<q; i++){
                    if(i<n)
                        recvcounts[i] = 1;
                    else
                        recvcounts[i] = 0;
                    displacement[i] = i;
                }
            }
            
        }

        MPI_Gatherv(local_vector, sendcount, MPI_DOUBLE, output_vector, recvcounts, displacement, MPI_DOUBLE, root, commcol);
        free(recvcounts);
        free(displacement); 
    }// 1st column ends
    MPI_Comm_free(&commcol); 
}


void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    // distribute the matrix which is initially stored in rank00
    // First step: To scatterv the rows to respective processors
    // Second step: To scatter in the columns
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int q = sqrt(comm_size);
    int floor_nq = n/q, ceil_nq = n/q + (n % q != 0);
    int rank00, myrank;
    int coords[2] = {0, 0};
    // getting the processor rank corresponding to the cart rank (0, 0)
    MPI_Cart_rank(comm, coords, &rank00);
    MPI_Comm_rank(comm, &myrank); // actual physical rank of processor
    
    // get block size for each of the local_matrix based on the cart ranks 
    // for every processor, get its cartesian coordinates in comm world
    MPI_Cart_coords(comm, myrank, 2, coords);

    // Creating 2 subcomms for row and cols
    MPI_Comm commcol, commrow;// subcomm consisting of columns
    int remain_dims[2] = {1, 0};// select the columns
    MPI_Cart_sub(comm, remain_dims, &commcol);
    remain_dims[0] = 0; remain_dims[1] = 1; // select the rows
    MPI_Cart_sub(comm, remain_dims, &commrow);

    double *temp_row = (double *)malloc(ceil_nq*n*sizeof(double));
    // STEP 1: distributing along rows in the 1st column
    if(coords[1]==0){
        // Finding the ranks in the particular column
        int commcol_rank, commcol_size;
        MPI_Comm_rank(commcol, &commcol_rank);
        MPI_Comm_size(commcol, &commcol_size);

        int recvcount, root;
        int *sendcounts = (int *)malloc(q*sizeof(int));
        int *displacement = (int *)malloc(q*sizeof(int));

        if(commcol_rank < n%q)
            recvcount = ceil_nq*n;
        else
            recvcount = floor_nq*n;

        root = rank00;
        if(myrank==rank00){
            displacement[0] = 0;
            if(n>=q){
                for(int i=0; i<q; i++){
                    if(i<n%q)
                        sendcounts[i] = ceil_nq*n;
                    else
                        sendcounts[i] = floor_nq*n;
                    if(i<q-1)
                        displacement[i+1] = displacement[i] + sendcounts[i];
                }
            }
            else{
                for(int i=0; i<q; i++){
                    if(i<n)
                        sendcounts[i] = 1*n;
                    else
                        sendcounts[i] = 0;
                    displacement[i] = i;
                }
            }
        }
        MPI_Scatterv(input_matrix, sendcounts, displacement, MPI_DOUBLE, temp_row, recvcount, MPI_DOUBLE, root, commcol);
        free(sendcounts);
        free(displacement);
    }// 1st column ends : so does STEP 1
    // STEP 2: distribute chunks of matrix along the rows using the rowcomms
    // note: we run for the commrows simulatneously, distribute data from the commrow_rank=0
    
    int commrow_rank, commrow_size;
    MPI_Comm_rank(commrow, &commrow_rank);
    MPI_Comm_size(commrow, &commrow_size);
    int recvcount2;
    int *sendcounts2 = (int *)malloc(q*sizeof(int));
    int *displacement2 = (int *)malloc(q*sizeof(int));

    if(commrow_rank==0){
        displacement2[0] = 0;
        if(n>=q){
            for(int i=0; i<q; i++){
                if(i<n%q)
                    sendcounts2[i] = ceil_nq;//*ceil_nq;
                else
                    sendcounts2[i] = floor_nq;//*ceil_nq;
                if(i<q-1)
                    displacement2[i+1] = displacement2[i] + sendcounts2[i];//ceil_nq;
            }
        }
        else{
            for(int i=0; i<q; i++){
                if(i<n)
                    sendcounts2[i] = 1*ceil_nq;
                else
                    sendcounts2[i] = 0;
                displacement2[i] = i;
            }
        }
    }

    // define the chunk type to single column
    // Note: there will be 2 different chunk sizes depending on processor rank < n%q
    if(coords[0]<n%q){// these processors will have ceil_nq number of rows
        if(commrow_rank<n%q)
            recvcount2 = ceil_nq;
        else
            recvcount2 = floor_nq;
        MPI_Datatype chunk_ceil_nq, local_chunk_ceil_nq;
        MPI_Type_vector(ceil_nq, 1, n, MPI_DOUBLE, &chunk_ceil_nq);
        MPI_Type_create_resized(chunk_ceil_nq, 0, sizeof(double), &chunk_ceil_nq);
        MPI_Type_commit(&chunk_ceil_nq);

        MPI_Type_vector(ceil_nq, 1, recvcount2, MPI_DOUBLE, &local_chunk_ceil_nq);
        MPI_Type_create_resized(local_chunk_ceil_nq, 0, sizeof(double), &local_chunk_ceil_nq);
        MPI_Type_commit(&local_chunk_ceil_nq);
        MPI_Scatterv(temp_row, sendcounts2, displacement2, chunk_ceil_nq, *local_matrix, recvcount2, local_chunk_ceil_nq, 0, commrow);

        MPI_Type_free(&local_chunk_ceil_nq);
        MPI_Type_free(&chunk_ceil_nq);
    }
    else{// coords[0]>=n%q // these processors will have floor_nq number of rows
        if(commrow_rank<n%q)
            recvcount2 = ceil_nq;// floor_nq*ceil_nq;
        else
            recvcount2 = floor_nq;
        MPI_Datatype chunk_floor_nq, local_chunk_floor_nq;
        MPI_Type_vector(floor_nq, 1, n, MPI_DOUBLE, &chunk_floor_nq);
        MPI_Type_create_resized(chunk_floor_nq, 0, sizeof(double), &chunk_floor_nq);
        MPI_Type_commit(&chunk_floor_nq);

        MPI_Type_vector(floor_nq, 1, recvcount2, MPI_DOUBLE, &local_chunk_floor_nq);// it is on the receving side.. can be changed!!!
        MPI_Type_create_resized(local_chunk_floor_nq, 0, sizeof(double), &local_chunk_floor_nq);
        MPI_Type_commit(&local_chunk_floor_nq);
        MPI_Scatterv(temp_row, sendcounts2, displacement2, chunk_floor_nq, *local_matrix, recvcount2, local_chunk_floor_nq, 0, commrow);

        MPI_Type_free(&local_chunk_floor_nq);
        MPI_Type_free(&chunk_floor_nq);
        free(sendcounts2);
        free(displacement2);
    }
    MPI_Comm_free(&commcol); 
    MPI_Comm_free(&commrow); 
    free(temp_row);
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int q = sqrt(comm_size);
    int floor_nq = n/q, ceil_nq = n/q + (n % q != 0);
    int myrank;
    int coords[2] = {0, 0};
    int temp_coords[2], source_rank, dest_rank, vector_size;
    // getting the processor rank corresponding to the cart rank (0, 0)
    MPI_Comm_rank(comm, &myrank); // actual physical rank of processor
    // get block size for each of the local_matrix based on the cart ranks 
    // for every processor, get its cartesian coordinates in comm world
    MPI_Cart_coords(comm, myrank, 2, coords);
    MPI_Status status;
    MPI_Comm commcol, commrow;// subcomm consisting of columns
    int remain_dims[2] = {1, 0};// select the columns
    MPI_Cart_sub(comm, remain_dims, &commcol);
    remain_dims[0] = 0; remain_dims[1] = 1; // select the rows
    MPI_Cart_sub(comm, remain_dims, &commrow);
    
    // setting the appropriate vector sizes
    // For each processor get it's distributing processor's rank to determine the vector size
    // decide vector size with the column size 
    if(n>=q){
        if(coords[1]<n%q)
            vector_size = ceil_nq;
        else
            vector_size = floor_nq;
    }
    else{
        if(coords[1]<n)
            vector_size = 1;
        else
            vector_size = 0;
    }

    if(coords[0]==0&&coords[1]==0){
        for(int i=0; i<vector_size; i++)
            row_vector[i] = col_vector[i];
    }

    if(coords[1]==0 && coords[0]!=0){// 1st column and skip the 00 block
        double send_vector_size;
        if(n>=q){
            if(coords[0]<n%q)
                send_vector_size = ceil_nq;
            else
                send_vector_size = floor_nq;
        }
        else{
            if(coords[0]<n)
                send_vector_size = 1;
            else
                send_vector_size = 0;
        }
        temp_coords[0] = coords[0];// j=i
        temp_coords[1] = coords[0];
        MPI_Cart_rank(comm, temp_coords, &dest_rank);
        MPI_Send(col_vector, send_vector_size, MPI_DOUBLE, dest_rank, 0, comm);
    }
    if(coords[0]==coords[1] && coords[0]!=0){// receive the vector
        temp_coords[0] = coords[0];
        temp_coords[1] = 0; // source processor
        MPI_Cart_rank(comm, temp_coords, &source_rank);
        MPI_Recv(row_vector, vector_size, MPI_DOUBLE, source_rank, 0, comm, &status);
    }
    // initiate a column subcomm and broadcast the value in the column
    temp_coords[0] = coords[1]; // set the column accordingly
    MPI_Cart_rank(commcol, temp_coords, &source_rank);
    MPI_Bcast(row_vector, vector_size, MPI_DOUBLE, source_rank, commcol);    
    MPI_Comm_free(&commcol); 
    MPI_Comm_free(&commrow); 
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
    // STEP 1: transpose and broadcast the vector 
    // STEP 2: Multiply locally and store
    // STEP 3: Do a parallel reduction (commrow)
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int q = sqrt(comm_size);
    int floor_nq = n/q, ceil_nq = n/q + (n % q != 0);
    double *row_vector = (double *)malloc(ceil_nq*sizeof(double));
    int coords[2] = {0, 0}, myrank;
    MPI_Comm_rank(comm, &myrank); // actual physical rank of processor
    // get block size for each of the local_matrix based on the cart ranks 
    // for every processor, get its cartesian coordinates in comm world
    MPI_Cart_coords(comm, myrank, 2, coords);
    transpose_bcast_vector(n, local_x, row_vector, comm);
    // Doing the local multiplication
    int Ax, Ay; // size of the local_A array
    if(coords[0]<n%q){// Ax~coords[1] and Ay~coords[0](rows)
        Ay = ceil_nq;
        if(coords[1]<n%q)
            Ax = ceil_nq;
        else
            Ax = floor_nq;
    }
    else{
        Ay = floor_nq;
        if(coords[1]<n%q)
            Ax = ceil_nq;
        else
            Ax = floor_nq;
    }
    
    double *local_results = (double *)malloc(Ay*sizeof(double));
    for(int i=0; i<Ay; i++){
        local_results[i] = 0;// initialise to zeros
        for(int j=0; j<Ax; j++){
            local_results[i] += local_A[i*Ax+j]*row_vector[j];
        }
    }
    // Do a MPI_Reduce parallely over commrow
    MPI_Comm commcol, commrow;// subcomm consisting of columns
    int remain_dims[2] = {1, 0};// select the columns
    MPI_Cart_sub(comm, remain_dims, &commcol);
    remain_dims[0] = 0; remain_dims[1] = 1; // select the rows
    MPI_Cart_sub(comm, remain_dims, &commrow);
    
    MPI_Reduce(local_results, local_y, Ay, MPI_DOUBLE, MPI_SUM, 0, commrow);
    MPI_Comm_free(&commcol); 
    MPI_Comm_free(&commrow);
    free(row_vector);
    free(local_results); 
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
    // Not taken care of the case n<p;
    // TODO : DON'T forget to implement the termination criterion. USE MPI_BARRIERS
    // STEP 1: Initialise the local_x-> 0
    // STEP 2: Start the loop.
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int q = sqrt(comm_size);
    int floor_nq = n/q, ceil_nq = n/q + (n % q != 0);
    int coords[2] = {0, 0}, myrank, rank00;
    // getting the processor rank corresponding to the cart rank (0, 0)
    MPI_Cart_rank(comm, coords, &rank00);
    MPI_Comm_rank(comm, &myrank); // actual physical rank of processor
    // get block size for each of the local_matrix based on the cart ranks 
    // for every processor, get its cartesian coordinates in comm world
    MPI_Cart_coords(comm, myrank, 2, coords);
    int Ax, Ay; // size of the local_A array
    if(n>=q){
        if(coords[0]<n%q){// Ax~coords[1] and Ay~coords[0](rows)
            Ay = ceil_nq;
            if(coords[1]<n%q)
                Ax = ceil_nq;
            else
                Ax = floor_nq;
        }
        else{
            Ay = floor_nq;
            if(coords[1]<n%q)
                Ax = ceil_nq;
            else
                Ax = floor_nq;
        }
    }
    else{
        if(coords[0]<n%q){// Ax~coords[1] and Ay~coords[0](rows)
            Ay = 1;
            if(coords[1]<n)
                Ax = 1;
            else
                Ax = 0;
        }
        else{
            Ay = 1;
            if(coords[1]<n)
                Ax = 1;
            else
                Ax = 0;
        }
    }
    double *local_R = (double *)malloc(Ax*Ay*sizeof(double));// local results

    int diag_size = Ay;
    if(Ax<Ay)
        diag_size = Ax;
    if(n>=q){
        if(coords[1]<n%q)
            diag_size = ceil_nq;
        else
            diag_size = floor_nq;
    }
    else{
        if(coords[1]<n)
            diag_size = 1;
        else
            diag_size = 0;
    }
    double *local_diag = (double *)malloc(diag_size*sizeof(double));

    for(int i=0; i<Ay; i++){
        for(int j=0; j<Ax; j++){
            local_R[i*Ax+j] = local_A[i*Ax+j];
            if(coords[0]==coords[1] && i==j){// diag processors
                local_R[i*Ax+i] = 0;
                local_diag[i] = local_A[i*Ax+j];
            }
        }
    }
    
    // SEND THE local_diagonals using the commrow
    MPI_Comm commcol, commrow;// subcomm consisting of columns
    int remain_dims[2] = {1, 0};// select the columns
    MPI_Cart_sub(comm, remain_dims, &commcol);
    remain_dims[0] = 0; remain_dims[1] = 1; // select the rows
    MPI_Cart_sub(comm, remain_dims, &commrow);
    int temp_coords[2], dest_rank, source_rank;
    MPI_Status status;

    double *diag_vector = (double *)malloc(Ay*sizeof(double));
    double *local_y = (double *)malloc(Ay*sizeof(double));
    double *Rx = (double *)malloc(Ay*sizeof(double));// Matrix R times x
    if(coords[1]==0){// 1st column
        for(int i=0; i<Ay; i++){// initialising the local_x on column1 as zeros
            local_x[i] = 0;// local_y[i] = 0;
        }
        if(coords[0]==0){// rank00
            for(int i=0; i<Ay; i++)
                diag_vector[i] = local_diag[i];
        }
    } 

    if(coords[0]==coords[1] && coords[0]!=0){// receive the diagonal values from diagonal processors
        temp_coords[0] = coords[0];
        temp_coords[1] = 0; // source processor
        MPI_Cart_rank(comm, temp_coords, &dest_rank);
        MPI_Send(local_diag, diag_size, MPI_DOUBLE, dest_rank, 0, comm);
    }
    if(coords[1]==0 && coords[0]!=0){// 1st column and skip the 00 block
        temp_coords[0] = coords[0];// j=i
        temp_coords[1] = coords[0];
        MPI_Cart_rank(comm, temp_coords, &source_rank);
        MPI_Recv(diag_vector, diag_size, MPI_DOUBLE, source_rank, 0, comm, &status);
    }
    

    double global_norm; int itr=0;
    double local_norm;
    // starting the Jacobi iteration 
    while(itr<max_iter){
        distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);
        if(coords[1]==0){
            local_norm = 0;
            // calculate norm(local_y, local_b) and do a reduction to rank00
            for(int i=0; i<Ay; i++){
                local_norm += (local_y[i]-local_b[i])*(local_y[i]-local_b[i]);
            }
            MPI_Reduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, rank00, commcol);
            if(myrank==rank00){
                global_norm = sqrt(global_norm);
            }
        }
        MPI_Bcast(&global_norm, 1, MPI_DOUBLE, rank00, comm);
        if(global_norm < l2_termination){
            if(myrank == rank00)
                printf("MPI_JACOBI: norm(Ax-b)<L in %d number of iterations\n", itr);
            break;
        }
        // Update the local_x value if not terminated
        distributed_matrix_vector_mult(n, local_R, local_x, Rx, comm);
        if(coords[1]==0){
            for(int i=0; i<Ay; i++){// Diag condition ==0 CHECK!!! Will not happen as we want diag > other elements
                local_x[i] = 1/diag_vector[i]*(local_b[i]-Rx[i]);
            }
/*
            for(int i=0; i<Ay; i++){
                printf("ITR = %d: GLOBAL_NORM = %lf and myrank[%d] and RX[%d] = %lf , local_x=%lf\n",itr, global_norm, myrank, i, Rx[i], local_x[i]);
            }
*/
        } 
        itr++;
//        MPI_Barrier(comm);
    }
    MPI_Comm_free(&commcol); 
    MPI_Comm_free(&commrow);
    free(local_R);
    free(local_diag); 
    free(diag_vector);
    free(local_y);
    free(Rx);
}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int q = sqrt(size);
    int floor_nq = n/q, ceil_nq = n/q + (n % q != 0);
    // distribute the array onto local processors!
    double* local_A = (double *)malloc(ceil_nq*ceil_nq*sizeof(double));
    double* local_x = (double *)malloc(ceil_nq*sizeof(double));

    my_get_grid_comm(&comm);
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);
    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);
    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
    free(local_y);
    free(local_A);
    free(local_x);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
//    MPI_Barrier(comm);
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int q = sqrt(size);
    // Special case: if n<q (we need to remove the additional processors)
    // create a new subcomm with q processors.
    if(n<q){
        int color = 0;
        if (rank<n*n){ color = 1;printf("process newcomm rank =%d and color = %d\n", rank, color);}
        MPI_Comm newcomm;
//        MPI_Comm_split(comm, rank<n*n, rank, &newcomm);
        MPI_Comm_split(comm, color, rank, &newcomm);
        if(color==0){return;}
       // comm = newcomm;
//        my_get_grid_comm(&newcomm);
    q = n;    
    int dims[2] = {q, q};
    int periods[2] = {0, 0};
    MPI_Cart_create(newcomm, 2, dims, periods, 0, &newcomm);
        MPI_Comm_rank(newcomm, &rank);
        MPI_Comm_size(newcomm, &size);
        q = sqrt(size);
        printf("n=%d and q=%d and size =%d\n", n, q, size);
        int floor_nq = n/q, ceil_nq = n/q + (n % q != 0);
        // distribute the array onto local processors!
        double* local_A = (double *)malloc(ceil_nq*ceil_nq*sizeof(double));
        double* local_b = (double *)malloc(ceil_nq*sizeof(double));
        distribute_matrix(n, &A[0], &local_A, newcomm);
        distribute_vector(n, &b[0], &local_b, newcomm);
        
        // allocate local result space
        double* local_x = new double[block_decompose_by_dim(n, newcomm, 0)];
        distributed_jacobi(n, local_A, local_b, local_x, newcomm, max_iter, l2_termination);
        // gather results back to rank 0
        gather_vector(n, local_x, x, newcomm);
        free(local_A);
        free(local_b);
    }
    else{
        int floor_nq = n/q, ceil_nq = n/q + (n % q != 0);
        // distribute the array onto local processors!
        double* local_A = (double *)malloc(ceil_nq*ceil_nq*sizeof(double));
        double* local_b = (double *)malloc(ceil_nq*sizeof(double));
        distribute_matrix(n, &A[0], &local_A, comm);
        distribute_vector(n, &b[0], &local_b, comm);
        
        // allocate local result space
        double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
        distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);
        // gather results back to rank 0
        gather_vector(n, local_x, x, comm);
        free(local_A);
        free(local_b);
    }
}
