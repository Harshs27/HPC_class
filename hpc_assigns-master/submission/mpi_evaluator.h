/*
 * CX 4220 / CSE 6220 Introduction to High Performance Computing
 *              Programming Assignment 1
 * 
 *  MPI polynomial evaluation algorithm function definitions
 * 
 */

/* 
 * File:   evaluator.h
 * Author: samindaw
 *
 * Created on February 17, 2017, 9:43 AM
 */

/*********************************************************************
 *                  !!  DO NOT CHANGE THIS FILE  !!                  *
 *********************************************************************/

#ifndef MPI_EVALUATOR_H
#define MPI_EVALUATOR_H

#include <mpi.h>

/**
 * 
 * Block decomposes an arrays of size n among processors
 * 
 * @param n                 Size of the array that needs to be scattered
 * @param scatter_values    Pointer to an array containing the data that needs 
 *                          scattering
 * @param n_local           The function will update this with the size of the 
 *                          local portion of the scattered array.
 * @param local_values      Pointer to an array containing the local portion of
 *                          the scattered array. 
 *                          Note: 
 *                              The function implementation should allocate  
 *                              memory to this array before saving any data to 
 *                              this {eg: using malloc(...)}
 * @param source_rank       Rank of the processor containing the scatter_values
 * @param comm              MPI communicator object
 */
void scatter(const int n, double* scatter_values, int &n_local, 
        double* &local_values, int source_rank, const MPI_Comm comm);

/**
 * 
 * Broadcast a single value among all other processors
 * 
 * @param value             Value to broadcast
 * @param source_rank       Processor containing the value to broadcast
 * @param comm              MPI communicator object
 * @return                  Value obtained through broadcast
 */
double broadcast(double value, int source_rank, const MPI_Comm comm);

/**
 * 
 * Perform parallel prefix algorithm on an array of n elements with SUM or 
 * PRODUCT operation
 * 
 * @param n                 Size of the array containing the values
 * @param values            Pointer to the array containing the values
 * @param prefix_results    Pointer to the array containing prefix sum result
 * @param OP                PREFIX_OP_SUM | PREFIX_OP_PRODUCT
 * @param comm              MPI communicator object
 */
void parallel_prefix(const int n, const double* values, double* prefix_results, const int OP, const MPI_Comm comm);

/**
 * 
 * Perform parallel polynomial evaluation given the value for x and the constants
 * 
 * @param x                 Value of x to be applied to the polynomial function
 * @param n                 No of constants
 * @param constants         Pointer to an array containing the constants
 * @param comm              MPI communicator object
 * @return                  Result of evaluating the polynomial function
 */
double mpi_poly_evaluator(const double x, const int n, const double* constants, const MPI_Comm comm);

#endif /* MPI_EVALUATOR_H */

