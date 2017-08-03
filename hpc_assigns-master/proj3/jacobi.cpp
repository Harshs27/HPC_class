/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
    for(int i=0; i<n; i++){
        y[i] = 0;
        for(int j=0; j<n; j++){
            y[i] += A[i*n+j]*x[j]; 
        }
    }
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    for(int i=0; i<n; i++){
        y[i] = 0;
        for(int j=0; j<m; j++){
            y[i] += A[i*m+j]*x[j];
        }
    }
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    double *update_x = (double *)malloc(n*sizeof(double));
    int itr=0;
    // So, we need to initialise all x to zeros..
    for(int i=0; i<n; i++)
        x[i] = 0;
    // for R, we can just zero out the elements of A 
    while(itr<max_iter){
        double total_norm = 0, Ax_i = 0, Rx_i=0;
        for(int i=0; i<n; i++){
            Ax_i = 0, Rx_i = 0;
            for(int j=0; j<n; j++){
                Ax_i += A[i*n+j]*x[j];
                if(i!=j){
                    Rx_i += A[i*n+j]*x[j];
                }
            }
            total_norm += (Ax_i - b[i])*(Ax_i-b[i]);// note: first norm is calculated
            update_x[i] = 1/A[i*n+i]*(b[i]-Rx_i); // then the x is updated for that row
        }
        // condition to terminate the iterations
        if(sqrt(total_norm)<l2_termination){
            printf("SERIAL JACOBI: norm(Ax-b)<L in %d number of iterations\n", itr);
            break;
        }
        // update the x
        for(int i=0; i<n; i++){
            x[i] = update_x[i];
//            printf("itr = %d Total_norm = %lf and x[%d]=%lf\n", itr, sqrt(total_norm), i, x[i]);
        }
        itr++;
    }
}
