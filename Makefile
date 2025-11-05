Q ?= 22
B ?= 256
lgH ?= 8
N ?= 100000000

CXX                     = nvcc -O3 -Wno-deprecated-gpu-targets
CUB                             = cub-1.8.0

DIR_RADIX               = ./gpu-radixsort
DIR_CUB                 = ./cub-radixsort
DIR_FUT                 = ./futhark-radixsort

SRC_RADIX               = $(DIR_RADIX)/radix-sort-gpu.cu
SRC_CUB                 = $(DIR_CUB)/radix-sort-cub.cu
SRC_FUT                 = $(DIR_FUT)/radix-sort-fut.fut

HELPERS                 = ./helper.h
KERNELS                 = $(DIR_RADIX)/kernels.cuh $(DIR_RADIX)/pbb_kernels.cuh

EXEC_RADIX              = radix-sort
EXEC_CUB                = cub-sort
EXEC_FUT                = fut-sort

NVCCFLAGS += -DQ_def=$(Q) -DB_def=$(B) -DlgH_def=$(lgH)

default: compile_cub run_cub run_fut compile_radix run_radix

compile_radix:  $(EXEC_RADIX)
compile_cub:    $(EXEC_CUB)
compile_fut:    $(EXEC_FUT)


$(EXEC_RADIX): $(SRC_RADIX) $(HELPERS) $(KERNELS)
	$(CXX) $(NVCCFLAGS) -o $(EXEC_RADIX) $(SRC_RADIX)

$(EXEC_CUB): $(SRC_CUB) $(HELPERS)
	$(CXX) -I$(CUB)/cub -o $(EXEC_CUB) $(SRC_CUB)

run_radix: $(EXEC_RADIX)
	./$(EXEC_RADIX) $(N)

run_cub: $(EXEC_CUB)
	./$(EXEC_CUB) $(N)

run_fut: $(SRC_FUT)
	futhark pkg add github.com/diku-dk/sorts
	futhark pkg sync
	futhark dataset --seed=2025 --u32-bounds=0:4294967295 -g [$(N)]u32 > $(DIR_FUT)/data.in
	futhark bench $(SRC_FUT)


validation_data:
	futhark dataset --seed=2025 --u32-bounds=0:4294967295 -g [500000]u32 > data.in

clean:
	rm -f $(EXEC_RADIX) $(EXEC_CUB) $(EXEC_FUT)
	rm -rf lib futhark.pkg
	rm  $(DIR_FUT)/data.in $(DIR_FUT)/radix-sort-fut  $(DIR_FUT)/radix-sort-fut.c


