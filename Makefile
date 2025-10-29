CXX        		= nvcc -O3
CUB		   		= cub-1.8.0

DIR_RADIX 		= ./gpu-radixsort
DIR_CUB			= ./cub-radixsort
DIR_FUT			= ./futhark-radixsort

SRC_RADIX 		= $(DIR_RADIX)/radix-sort-gpu.cu
SRC_CUB			= $(DIR_CUB)/radix-sort-cub.cu
SRC_FUT			= $(DIR_FUT)/radix-sort-fut.fut

HELPERS			= ./helper.h
KERNELS 		= $(DIR_RADIX)/kernels.cuh $(DIR_RADIX)/pbb_kernels.cuh

EXEC_RADIX		= radix-sort
EXEC_CUB		= cub-sort
EXEC_FUT		= fut-sort

-- compiled input @ data.in
-- output @ sorted-data.out


default: compile_cub run_cub compile_radix run_radix

compile_radix: 	$(EXEC_RADIX)
compile_cub: 	$(EXEC_CUB)
compile_fut:	$(EXEC_FUT)


$(EXEC_RADIX): $(SRC_RADIX) $(HELPERS) $(KERNELS)
	$(CXX) -o $(EXEC_RADIX) $(SRC_RADIX)

$(EXEC_CUB): $(SRC_CUB) $(HELPERS)
	$(CXX) -I$(CUB)/cub -o $(EXEC_CUB) $(SRC_CUB)

$(EXEC_FUT): $(SRC_FUT)
	futhark cuda -o $(EXEC_FUT) $(SRC_FUT)

run_radix: $(EXEC_RADIX)
	./$(EXEC_RADIX) 100000000

run_cub: $(EXEC_CUB)
	./$(EXEC_CUB) 100000000

# run_fut: $(EXEC_FUT)



validation_data: 
	futhark dataset --seed=2025 --u32-bounds=0:4294967295 -g [500000]u32 > data.in

clean:
	rm -f $(EXEC_RADIX) $(EXEC_CUB)