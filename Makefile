CXX        		= nvcc -O3
CUB		   		= cub-1.8.0

DIR_RADIX 		= ./gpu-radixsort
DIR_CUB			= ./cub-radixsort
DIR_FUTHARK		= ./futhark-radixsort

SRC_RADIX 		= $(DIR_RADIX)/radix-sort-gpu.cu
SRC_CUB			= $(DIR_CUB)/radix-sort-cub.cu
HELPERS			= ./helper.h
KERNELS 		= $(DIR_RADIX)/kernels.cuh $(DIR_RADIX)/pbb_kernels.cuh
EXEC_RADIX		= radix-sort
EXEC_CUB		= cub-sort


default: compile_cub run_cub compile_radix run_radix

compile_radix: 	$(EXEC_RADIX)
compile_cub: 	$(EXEC_CUB)

$(EXEC_RADIX): $(SRC_RADIX) $(HELPERS) $(KERNELS)
	$(CXX) -o $(EXEC_RADIX) $(SRC_RADIX)

$(EXEC_CUB): $(SRC_CUB) $(HELPERS)
	$(CXX) -I$(CUB)/cub -o $(EXEC_CUB) $(SRC_CUB)

run_radix: $(EXEC_RADIX)
	./$(EXEC_RADIX) 100000000

run_cub: $(EXEC_CUB)
	./$(EXEC_CUB) 100000000

clean:
	rm -f $(EXEC_RADIX) $(EXEC_CUB)