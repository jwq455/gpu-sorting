# Implementation of fast sorting algorithms on GPU

### Authors: Johannes Rosendal, Helene Møller-Jensen & Oscar Halvoring Augustinus

This project is part of the course `Programming Massively Parallel Hardware` at the department of Computer Science at University of Copenhagen (UCPH).  

Our implementation of radix sort is compiled and run using the Makefile. 
Running `make` (recommended running in silent mode `--silent`) will compile radix sort for GPU with default parameters B=256, Q=22, lgH=8 and N=100000000 and run the resulting executable along with CUP and and futharks radixsort on the same number of elements for comparison. Note this will take a while because of compile time and futhark. 

You can compile radix sort with different parameters by defining them when running make. For example
`make N=100000 B=512`
