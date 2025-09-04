# ---- Compilers & flags ----
CC = mpicc
NVCC     ?= nvcc

# Toggle GPU build: make USE_CUDA=0 to disable
USE_CUDA ?= 1

CFLAGS   ?= -O3 -std=c11 -fopenmp -Wall -Wextra -Wno-sign-compare
LDFLAGS  ?= -fopenmp
CUDA_HOME ?= /usr/local/cuda
# Set a reasonable default arch if you want (commented to stay portable)
# CUDA_ARCH ?= -arch=sm_70

# ---- Outputs ----
BIN_DIR = build
TARGET  = $(BIN_DIR)/pds_project_mpi_omp_c

# ---- Sources ----
SRCS_C   = src/main.c src/compute.c src/io.c
OBJS_C   = $(SRCS_C:.c=.o)

ifeq ($(USE_CUDA),1)
  SRCS_CU  = src/cuda_match.cu
  OBJS_CU  = $(SRCS_CU:.cu=.o)
  CFLAGS  += -DUSE_CUDA
  LDLIBS  += -L$(CUDA_HOME)/lib64 -lcudart
else
  SRCS_CU  =
  OBJS_CU  =
endif

OBJS = $(OBJS_C) $(OBJS_CU)

# ---- Build rules ----
.PHONY: all clean

all: $(TARGET)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(TARGET): $(BIN_DIR) $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

# C sources
src/%.o: src/%.c src/types.h
	$(CC) $(CFLAGS) -c -o $@ $<

# CUDA sources (only if USE_CUDA=1)
# Pass OpenMP & warnings to host compiler; keep arch generic unless you know your GPUs
src/%.o: src/%.cu src/types.h
	$(NVCC) -O3 $(CUDA_ARCH) -Xcompiler="-fopenmp -Wall -Wextra" -c -o $@ $<

clean:
	rm -rf $(BIN_DIR) src/*.o