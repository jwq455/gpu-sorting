# Implementation of fast sorting algorithms on GPU

### Authors: Johannes Rosendal, Helene Møller-Jensen & Oscar Halvoring Augustinus

This project is part of the course `Programming Massively Parallel Hardware` at the department of Computer Science at University of Copenhagen (UCPH).  


Our implementation of radix sort is compiled and run using the Makefile. 
Running make will compile radix sort for GPU with default parameters B=256, Q=22, lgH=8 and N=100000000 and run the resulting executable along with CUP on the same number of elements for comparison. 

You can compile radix sort with different parameters by defining them when running make. 

The scripts bench.sh runs radix sort with various parameters and saves the result in benches.txt
