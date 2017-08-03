/*
 * CX 4220 / CSE 6220 Introduction to High Performance Computing
 *              Programming Assignment 1
 * 
 *  MPI polynomial evaluation algorithm function implementations go here
 * 
 */

#include "mpi_evaluator.h"
#include "const.h"

#include <math.h>

inline int posmod(int i, int n);

/* Distribute an array of size n across p processors with any 
   unevenness distrubteed across the first n%p ranks.
   */
//TODO: Decide between barrier and blocking send
void scatter(const int n, double* scatter_values, int &n_local, double* &local_values, int source_rank, const MPI_Comm comm){
    int p, rank;
    MPI_Comm_size(comm, &p); 
    MPI_Comm_rank(comm, &rank); 
    MPI_Status stat; 

    if(rank==source_rank){// scatter the data from this processor
        MPI_Request req; // Dummy Variable
        int message_size;
        int floor_np = n/p, ceil_np = n/p + (n%p != 0), extra = n%p;
        int idx = 0; //index for the increment pointer in source processor

        // The first `extra` processors get `ceil_np` values, and the rest get `floor_np`. 
        for(int i=0; i<p; i++){
            message_size = (i < extra) ? ceil_np : floor_np;

            // Just copy the data to ourself
            if(i==source_rank){
                local_values = (double *)malloc(message_size*sizeof(double));
                memcpy(local_values, scatter_values+idx, message_size*sizeof(double)); // Don't send to ourself
                n_local = message_size;
            }

            // Send data
            else{
                MPI_Isend(&message_size, 1, MPI_INT, i, 1, comm, &req);                      // Send number of values
                MPI_Isend(scatter_values+idx, message_size, MPI_DOUBLE, i, 1, comm, &req);   // Send values
            }

            idx += message_size; //The next send starts where this one ended
        }

<<<<<<< HEAD
        MPI_Request_free(&req);
    }
    else{// receive the data - we are not the source rank
        MPI_Recv(&n_local, 1, MPI_INT, source_rank, 1, comm, &stat);              // receive number of values
        local_values = (double *)malloc(n_local*sizeof(double));
        MPI_Recv(local_values, n_local, MPI_DOUBLE, source_rank, 1, comm, &stat); // receive values
    }
    MPI_Barrier(comm);
=======
    MPI_Request_free(&req);
  }
  else{// receive the data - we are not the source rank
    MPI_Recv(&n_local, 1, MPI_INT, source_rank, 1, comm, &stat);              // receive number of values
    local_values = (double *)malloc(n_local*sizeof(double));
    MPI_Recv(local_values, n_local, MPI_DOUBLE, source_rank, 1, comm, &stat); // receive values
  }
  
  MPI_Barrier(comm);
>>>>>>> 3f7febcd87b021e37e7691accc5717c4fd260524
}

/* Broadcast a value `value` from `source_rank` to all other ranks */
double broadcast(double value, int source_rank, const MPI_Comm comm){
    int p, rank;
    MPI_Status stat;
    int rank_c;
    MPI_Comm_size(comm, &p); 
    MPI_Comm_rank(comm, &rank); 

    rank = posmod(rank - source_rank, p); // shift ranks so source_rank is 0

    int d = ceil(log2(p)); //smallest d s.t. 2^d >= p

    for(int i=0; i<d; i++){// going from LSB to MSB
<<<<<<< HEAD
        rank_c = rank ^ (1 << i); //complement rank for communication
        // send and receive only when the rank is < p
        if(rank_c < p){
            if(rank < rank_c){
                MPI_Send(&value, 1, MPI_DOUBLE, posmod(rank_c+source_rank, p), 1, comm);
            }
            else{
                MPI_Recv(&value, 1, MPI_DOUBLE, posmod(rank_c+source_rank, p), 1, comm, &stat);
            }
        }
=======
      rank_c = rank ^ (1 << i); //complement rank for communication
      // send and receive only when the rank is < p
      if(rank_c < p){
	if(rank < rank_c && rank < (1<<i)){
	  //printf("%d: %d sends to %d\n", i, rank, rank_c);
	  MPI_Send(&value, 1, MPI_DOUBLE, posmod(rank_c+source_rank, p), 1, comm);
	}else if(rank_c < (1<<i)){
	  MPI_Recv(&value, 1, MPI_DOUBLE, posmod(rank_c+source_rank, p), 1, comm, &stat);
	}

	/*
        if(rank < rank_c && rank < (1<<i)){
	  //printf("%d: %d sends to %d\n", i, rank, rank_c);
	  MPI_Send(&value, 1, MPI_DOUBLE, posmod(rank_c+source_rank, p), 1, comm);
        }
        else{
	  MPI_Recv(&value, 1, MPI_DOUBLE, posmod(rank_c+source_rank, p), 1, comm, &stat);
        }*/

      }
>>>>>>> 3f7febcd87b021e37e7691accc5717c4fd260524
    }

    return value;
}

/* Choose which operator to use */
double my_op_function(double a, double b, const int OP){
    switch(OP){
        case PREFIX_OP_SUM:
            return a+b;
        case PREFIX_OP_PRODUCT:
            return a*b;
        default:
            printf("Error: Invalid Operator\nExiting...\n");
            exit(1);
    }
}

/* Perform Prefix-OP over p processors */
void parallel_prefix(const int n, const double* values, double* prefix_results, const int OP, const MPI_Comm comm){

    //Implementation
    double received, total;
    prefix_results[0] = values[0]; 
    for(int i = 1; i < n; i++){
        prefix_results[i] = my_op_function(prefix_results[i-1], values[i], OP);
    }
    total = prefix_results[n-1];

    int p, rank;
    MPI_Comm_size(comm, &p); // total number of processors.
    MPI_Comm_rank(comm, &rank); // rank of current processor
    MPI_Status stat;

    int d = ceil(log2(p)), rank_c;
    double accum = (OP == PREFIX_OP_SUM)? 0. : 1.;
    MPI_Request req;
    for(int i=0; i<d; i++){
        rank_c = rank ^ (1 << i);

        // Only communicate if complimentary rank actually exists
        if(rank_c < p){
            MPI_Irecv(&received, 1, MPI_DOUBLE, rank_c, 1, comm, &req);
            MPI_Send(&total, 1, MPI_DOUBLE, rank_c, 1, comm);
            MPI_Wait(&req, &stat);

            total = my_op_function(received, total, OP);// Apply OP to received and total
            if(rank > rank_c){
                accum = my_op_function(accum, received, OP);
            }
        }
    }

    for(int i = 0; i < n; i++){
        prefix_results[i] = my_op_function(prefix_results[i], accum, OP);
    }

    return;
}



double mpi_poly_evaluator(const double x, const int n, const double* constants, const MPI_Comm comm){
    //Implementation
    // Calculate the local prefix mul first.[x, x^2, x^3]
    // Run parallel prefix and get the prefix results
    // Again a local scan to update the results locally for the polynomial
    // A reduction over all the processors (use parallel_prefix with sum as operator)
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);
    MPI_Status stat;

    double *values         = (double *)malloc(n * sizeof(double)); // array [1, x, x, ..., x]
    double *temp           = (double *)malloc(n * sizeof(double)); // temp varaible
    double *poly_results   = (double *)malloc(n * sizeof(double)); // 

    // Prefix-Multiply over the array [1, x, x, ..., x] to calculate the powers of x in the polynomial
    values[0] = (rank == 0) ? 1. : x;
    for(int i = 1; i < n; i++){
        values[i] = x;
    }
    parallel_prefix(n, values, temp, PREFIX_OP_PRODUCT, comm); 

    // Evaluate our portion of the polynomial
    for(int i = 0; i < n; i++){
        temp[i] = constants[i] * temp[i]; 
    }

    // Prefix-Sum over the partial results to get the final answer
    parallel_prefix(n, temp, poly_results, PREFIX_OP_SUM, comm);

    // Send the answer to rank 0 and we're done
    double final_result = poly_results[n-1];

    if(p > 1){
        if(rank==p-1){
            MPI_Send(&final_result, 1, MPI_DOUBLE, 0, 1, comm);
        }
        if(rank>0){
            return 0.;
        }
        else{
            MPI_Recv(&final_result, 1, MPI_DOUBLE, p-1, 1, comm, &stat);
            return final_result;
        }
    }else{
        return final_result;
    }
}

/* Returns positive modulo */
inline int posmod(int i, int n) {
    return (i % n + n) % n;
}
