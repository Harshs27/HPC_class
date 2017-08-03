/*
 * CX 4220 / CSE 6220 Introduction to High Performance Computing
 *              Programming Assignment 1
 * 
 *  Serial polynomial evaluation algorithm function implementations goes here
 * 
 */
#include <stdio.h>
#include <iostream>
using namespace std;

double poly_evaluator(const double x, const int n, const double* constants){
    //Implementation
    double value = constants[0];
    double x0 = x;
    for(int i=1; i<n; i++){
//    cout << value << endl;
        value += constants[i]*x0;
        x0 *= x;
    }
    return value;
}
