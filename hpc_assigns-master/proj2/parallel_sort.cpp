/**
 * @file    parallel_sort.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements the parallel, distributed sorting function.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "parallel_sort.h"
#include <string.h>

// Recommended use case #define DEBUG if(level == 1) to get output from level 1 of the recursion (1 is the 1st level, not 0)
#define DEBUG if(0)

// This is the amount of data rank i will have to work on after the recursive call
#define SIZE(rank) ((rank < r)? ((rank < extra_leq) ? ceil_leq : floor_leq) : ((rank-r < extra_gt) ? ceil_gt : floor_gt))

// implementation of your parallel sorting
void parallel_sort(int * begin, int* end, MPI_Comm comm) {
  int rank, size;
  int pivot, pivot_rank;
  int leq, gt;
  int r, s;
  int n_local = (end - begin);
  int m[] = {0, 0}, *m_recv;
  float split_fraction;
  MPI_Comm newcomm;
  static int level = 0;
  level++;
   
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // Run serial quick-sort 
  if(size == 1){
    qsort(begin, n_local, sizeof(int), cmp_int);
    level--;
    return;
  }

  DEBUG print_arr("data", begin, end-begin, rank);

  m_recv = (int*)malloc(sizeof(int) * 2 * size);

  // Pick a rank to pick a pivot
  srand(size);
  pivot_rank = rand() % size;

  // pivot_rank picks pivot and broadcasts
  pivot = begin[rand() % n_local]; 
  MPI_Bcast(&pivot, 1, MPI_INT, pivot_rank, comm);

  DEBUG printf("(%d) pivot: %d\n", rank, pivot);

  // m[0] Contains the number of local elements <= pivot
  // m[1] contains number of local elements > pivot
  for(int i = 0; i < n_local; i++)
    m[0] += (begin[i] <= pivot)? 1 : 0;
  m[1] = n_local - m[0];

  //TODO: Can this be done in place? 
  int *send_buf = (int*)malloc(sizeof(int) * n_local);
  int *leqp = send_buf;
  int *gtp = send_buf+m[0];
  for(int i = 0; i < n_local; i++){
    if(begin[i] <= pivot){
      *leqp++ = begin[i];
    }else{
      *gtp++ = begin[i];
    }
  }

  // Gather everyone's values for m into m_recv
  MPI_Allgather(m, 2, MPI_INT, m_recv, 2, MPI_INT, comm);

  //DEBUG print_arr("m_recv", m_recv, 2*size, rank);

  // leq contains the number of elements <= pivot across all ranks
  // gt  contains the number of elements >  pivot across all ranks
  leq = 0; gt = 0;
  for(int i = 0; i < 2*size; i+=2){
    leq += m_recv[i  ];
    gt  += m_recv[i+1];
  }
  
  //DEBUG printf("(%d) leq: %d, gt: %d\n", rank, leq, gt);

  // Decide how to split up the communicator 
  // The first r ranks work on leq data, the rest (s) work on gt data
  // Neither r nor s will ever be 0, even if leq or gt is 0
  split_fraction = (float)(leq) / (float)(leq + gt);
  r = (split_fraction * size) >= 1? (int) (split_fraction * size) : 1;
  s = size - r;
  if(s == 0){
    s = 1;
    r--;
  }

  //DEBUG printf("(%d) r: %d, s: %d\n", rank, r, s);

  // These are used for computing the sizes of the various ranks' workloads
  int floor_leq = leq / r, ceil_leq = leq/r + (leq % r != 0), extra_leq = leq % r;
  int floor_gt  = gt  / s, ceil_gt  = gt/s  + (gt  % s != 0), extra_gt  = gt  % s;

  //DEBUG printf("(%d) floor_leq: %d, ceil_leq: %d, extra_leq: %d\n", rank, floor_leq, ceil_leq, extra_leq);
  //DEBUG printf("(%d) floor_gt: %d, ceil_gt: %d, extra_gt: %d\n", rank, floor_gt, ceil_gt, extra_gt);

  // index of start/end of global leq/gt arrays we own 
  // we own [local_leq_s, local_leq_e) of the global [0, leq)
  // and    [local_gt_s, local_gt_e)   of the global [0, gt)
  int local_leq_s = 0, local_leq_e; 
  int local_gt_s = 0, local_gt_e;

  for(int i = 0; i < 2*rank; i+= 2){
    local_leq_s += m_recv[i];
    local_gt_s += m_recv[i+1];
  }
  local_leq_e = local_leq_s + m[0];
  local_gt_e = local_gt_s + m[1];

  //DEBUG printf("(%d) local_leq_s: %d, local_leq_e: %d\n", rank, local_leq_s, local_leq_e);
  //DEBUG printf("(%d) local_gt_s: %d, local_gt_e: %d\n", rank, local_gt_s, local_gt_e);

  int recv_s = 0, recv_e = 0;

  // Our rank will take care of either a portion of leq or a portion of gt
  // The range [recv_s, recv_e) is the portion of the global 
  // leq or gt array that we have been assigned
  if(rank < r){
    for(int i = 0; i < rank; i++){
      recv_s += SIZE(i);
    }
    recv_e = recv_s + SIZE(rank);
  }else{
    for(int i = r; i < rank; i++){
      recv_s += SIZE(i);
    }
    recv_e = recv_s + SIZE(rank);
  }

  // The number of elements we will have after the split
  int newsize = SIZE(rank);

  int *sendcounts = (int*)calloc(sizeof(int), size);
  int *recvcounts = (int*)calloc(sizeof(int), size);
  int *offsets    = (int*)calloc(sizeof(int), size);
  int *recvoff    = (int*)calloc(sizeof(int), size);
  int *recvbuf    = (int*)calloc(sizeof(int), newsize);

  int data_sent = 0;
  int leq_s = 0, leq_e, gt_s = 0, gt_e; //represent bounds of leq of current proc in iteration

  //DEBUG if(rank == 2) printf("(%d) we have [%d,%d) in leq \n", rank, local_leq_s, local_leq_e);

  //calculate sending counts, offsets stuff
  for(int i = 0; i < size; i++){
    if(i < r){
      int s, e;
      leq_e = leq_s + SIZE(i);
      
      e = min(leq_e, local_leq_e);
      s = max(leq_s, local_leq_s);

      sendcounts[i] = (e-s) > 0 ? (e-s) : 0;
      offsets[i] = data_sent;
      data_sent += sendcounts[i];

      leq_s = leq_e;
    }else{
      int s, e;
      gt_e = gt_s + SIZE(i);

      e = min(gt_e, local_gt_e);
      s = max(gt_s, local_gt_s);

      sendcounts[i] = (e-s) > 0 ? (e-s) : 0;
      offsets[i] = data_sent;
      data_sent += sendcounts[i];

      gt_s = gt_e;

    }
  }

  DEBUG printf("(%d) recv_s: %d, recv_e: %d\n", rank, recv_s, recv_e);

  //calculate recieving counts / offsets
  int owned_leq_s = 0, owned_leq_e = 0;
  int owned_gt_s = 0, owned_gt_e = 0;
  int data_recv = 0;
  
  if(rank < r){
    for(int i = 0; i < size; i++){
      int s, e;
      owned_leq_e += m_recv[2*i];
      
      //Calculate interval overlap
      e = min(recv_e, owned_leq_e);
      s = max(recv_s, owned_leq_s);

      recvcounts[i] = (e-s) > 0 ? (e-s) : 0;
      recvoff[i] = data_recv;
      data_recv += recvcounts[i];

      owned_leq_s = owned_leq_e;
    }
  }else{
    for(int i = 0; i < size; i++){
      int s, e;
      owned_gt_e += m_recv[2*i+1];

      //Calculate interval overlap
      e = min(recv_e, owned_gt_e);
      s = max(recv_s, owned_gt_s);

      recvcounts[i] = (e-s) > 0 ? (e-s) : 0;
      recvoff[i] = data_recv;
      data_recv += recvcounts[i];

      owned_gt_s = owned_gt_e;
    }

  }
    
  DEBUG print_arr("sendcounts", sendcounts, size, rank);
  DEBUG print_arr("offsets", offsets, size, rank);

  DEBUG print_arr("recvcounts", recvcounts, size, rank);
  DEBUG print_arr("recvoff", recvoff, size, rank);
  
  MPI_Alltoallv(send_buf, sendcounts, offsets, MPI_INT, 
		recvbuf, recvcounts, recvoff, MPI_INT, comm);
  
  //DEBUG printf("(%d) newsize: %d\n", rank, newsize);
  //DEBUG print_arr("recvbuf", recvbuf, newsize, rank);
  

  MPI_Comm_split(comm, rank < r, rank, &newcomm);
  free(send_buf);
  parallel_sort(recvbuf, recvbuf + newsize, newcomm);

  //DEBUG print_arr("sorted", recvbuf, newsize, rank);

  int want_s = 0, want_e;
  int have_s = 0, have_e;

  for(int i = 0; i < rank; i++){
    want_s += m_recv[2*i] + m_recv[2*i+1];
    have_s += SIZE(i);
  }
  want_e = want_s + n_local;
  have_e = have_s + newsize;

  //DEBUG printf("(%d) have: [%d,%d)\n", rank, have_s, have_e);
  //DEBUG printf("(%d) want: [%d,%d)\n", rank, want_s, want_e);

  memset(sendcounts, 0, sizeof(int) * size);
  memset(offsets,    0, sizeof(int) * size);
  memset(recvcounts, 0, sizeof(int) * size);
  memset(recvoff,    0, sizeof(int) * size);

  // Computes interval overlap for send and recv all in one loop
  int r_h_s = 0, r_h_e = 0, r_w_s = 0, r_w_e = 0; //stands for remote_[have/want]_[start/end]
  int sent_d = 0, recv_d = 0;
  for(int i = 0; i < size; i++){
    r_h_e += SIZE(i);
    r_w_e += m_recv[2*i] + m_recv[2*i+1];

    int ss = 0, se = 0, rs = 0, re = 0;

    //Send
    se = min(have_e, r_w_e);
    ss = max(have_s, r_w_s);
    
    sendcounts[i] = (se - ss) > 0 ? (se - ss) : 0;
    offsets[i] = sent_d;
    sent_d += sendcounts[i];
    
    //Recv
    rs = max(want_s, r_h_s);
    re = min(want_e, r_h_e);
    
    recvcounts[i] = (re - rs) > 0 ? (re - rs) : 0;
    recvoff[i] = recv_d;
    recv_d += recvcounts[i];

    r_h_s = r_h_e;
    r_w_s = r_w_e;
  }

  /*
  DEBUG print_arr("sendcounts2", sendcounts, size, rank);
  DEBUG print_arr("offsets2", offsets, size, rank);
  DEBUG print_arr("recvcounts2", recvcounts, size, rank);
  DEBUG print_arr("recvoff2", recvoff, size, rank);


  DEBUG print_arr("Redistributing", recvbuf, newsize, rank);
  */

  MPI_Alltoallv(recvbuf, sendcounts, offsets, MPI_INT,
      begin, recvcounts, recvoff, MPI_INT, comm );

  //DEBUG print_arr("Redis'd", begin, n, rank);

  //DEBUG print_arr("returning", begin, n_local, rank);
  level--;

  free(sendcounts);
  free(recvcounts);
  free(offsets);
  free(recvoff);
  free(recvbuf);
  free(m_recv);
  

  return;
}

/*
  Helper Functions
*/

int cmp_int(const void * a, const void * b){
  return ( *(int*)a - *(int*)b );
}

void print_arr(const char* name, int *a, int len, int rank){
  printf("(%d) %s: ", rank, name);
  for(int i = 0; i < len; i++){
    printf("%d ", a[i]);
  }
  printf("\n");
}
