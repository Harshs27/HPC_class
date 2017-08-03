# Makefile for HPC 6220 Programming Assignment 1
CXX=mpic++

# activate for compiler optimizations:
CCFLAGS=-Wall -O3 -std=c++0x
LDFLAGS=

all: poly-eval

poly-eval: main.o evaluator.o mpi_evaluator.o utils.o
	$(CXX) $(LDFLAGS) -o $@ $^


%.o: %.cpp %.h
	$(CXX) $(CCFLAGS) -c $<

%.o: %.cpp
	$(CXX) $(CCFLAGS) -c $<

run_serial:
	./poly-eval sample-constants.txt sample-values.txt

run_parallel:
	mpirun -np 5 ./poly-eval sample-constants.txt sample-values.txt

clean:
	rm -f *.o poly-eval test

test: main2.o evaluator.o mpi_evaluator.o utils.o
	$(CXX) $(LDFLAGS) -o $@ $^