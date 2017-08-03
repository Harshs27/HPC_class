/**
 * @file    mpi_tests.cpp
 * @ingroup group
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   GTest Unit Tests for the parallel MPI code.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
/*
 * Add your own test cases here. We will test your final submission using
 * a more extensive tests suite. Make sure your code works for many different
 * input cases.
 *
 * Note:
 * The google test framework is configured, such that
 * only errors from the processor with rank = 0 are shown.
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include "io.h"
#include "parallel_sort.h"

/*********************************************************************
 *                   Add your own test cases here.                   *
 *********************************************************************/
// Other test cases can include:
// - all elements are equal
// - elements are randomly picked
// - elements are sorted inversely
// - number of elements is not divisible by the number of processors
// - number of elements is smaller than the number of processors


// test parallel MPI matrix vector multiplication
TEST(MpiTest, Sort10)
{
//    int x_in[10] = {4, 7, 5, 1, 0, 2, 9, 3, 8, 6};
//    int y_ex[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // Read an input.txt file generated from input.py, store the array in x, store the sorted array in y
    FILE* f = fopen("input.txt", "r");
    int number = 0;
    int line_num = 0;
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    int size = 0;
    int *x_in, count=0;
    while((read=getline(&line, &len, f)) != -1){
        if(line_num == 0){
            size = atoi(line);
            printf("%d\n", size);
            line_num += 1;
        }
        else{ 
            if(line_num==1)
                x_in = (int *)malloc(size*sizeof(int));
            number = atoi(line);
            x_in[count] = number; count++;
            line_num += 1;  
        }
    }
    printf("inside mpi_tests\n");
    int L = size;
/*    for(int i=0; i<L;i++)
        printf(" %d\t", x_in[i]);
    printf("\n");
*/
    int *y_ex = (int *)malloc(L*sizeof(int));
    for(int i=0; i<L;i++)
        y_ex[i] = x_in[i];
    qsort(y_ex, L, sizeof(int), cmp_int);
    printf("sorted input for y");
/*    for(int i=0; i<L;i++)
        printf(" %d\t", y_ex[i]);
    printf("\n");
*/
    std::vector<int> x(x_in, x_in+L);
    std::vector<int> local_x = scatter_vector_block_decomp(x, MPI_COMM_WORLD);// scatter the data initially

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double t_start, t_end;
    t_start = MPI_Wtime();
    parallel_sort(&local_x[0], &local_x[0]+local_x.size(), MPI_COMM_WORLD);
//    MPI_BARRIER(MPI_COMM_WORLD);
    t_end = MPI_Wtime();
    double time_secs= (t_end - t_start);
    if (rank==0){
        printf("Runtime for sorting is secs = %lf \n", time_secs);
    }
    std::vector<int> y = gather_vectors(local_x, MPI_COMM_WORLD);

    if (rank == 0)
        for (int i = 0; i < L; ++i) {
            EXPECT_EQ(y_ex[i], y[i]);
        }
}
/*
int cmp_int(const void * a, const void * b){
      return ( *(int*)a - *(int*)b );
}
*/
