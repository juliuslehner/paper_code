# Compiler and flags
NVCC := nvcc
COMMON_CXXFLAGS := -std=c++17 -arch sm_80

# Include directories
INCLUDE_DIRS := -I/u/home/lehnj/nvidia-mathdx-24.01.0/nvidia/mathdx/24.01/include/ \
                -I/usr/local/cuda/include

# Library directories and libraries
LIB_DIRS := -L/usr/local/cuda/lib64
LIBS := -lnvidia-ml -lfftw3l -lfftw3 -lm

# Source files and their corresponding targets
SRCS_1D := block_fft_precision_many.cu block_fft_performance_many.cu
SRCS_2D := block_fft_precision_many_2d.cu block_fft_2d_performance_many_2d.cu

TARGETS_1D_PRECISION := block_fft_1d_precision_many_double block_fft_1d_precision_many_single block_fft_1d_precision_many_half
TARGETS_1D_PERFORMANCE := block_fft_1d_performance_many_double block_fft_1d_performance_many_single block_fft_1d_performance_many_half
TARGETS_2D_PRECISION := block_fft_2d_precision_many_double block_fft_2d_precision_many_single
TARGETS_2D_PERFORMANCE := block_fft_2d_performance_many_double block_fft_2d_performance_many_single 

TARGETS_1D := $(TARGETS_1D_PRECISION) $(TARGETS_1D_PERFORMANCE)
TARGETS_2D := $(TARGETS_2D_PRECISION) $(TARGETS_2D_PERFORMANCE)
TARGETS := $(TARGETS_1D) $(TARGETS_2D)

# Precision-specific compiler flags
DOUBLE_FLAGS := -DDOUBLE_PRECISION
SINGLE_FLAGS := -DSINGLE_PRECISION
HALF_FLAGS := -DHALF_PRECISION

# Default target
all: $(TARGETS)

# Rules for 1D FFT different precisions
block_fft_1d_precision_many_double: block_fft_precision_many.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(DOUBLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

block_fft_1d_precision_many_single: block_fft_precision_many.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(SINGLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

block_fft_1d_precision_many_half: block_fft_precision_many.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(HALF_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

block_fft_1d_performance_many_double: block_fft_performance_many.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(DOUBLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

block_fft_1d_performance_many_single: block_fft_performance_many.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(SINGLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

block_fft_1d_performance_many_half: block_fft_performance_many.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(HALF_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

# Rules for 2D FFT different precisions
block_fft_2d_precision_many_double: block_fft_precision_many_2d.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(DOUBLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

block_fft_2d_precision_many_single: block_fft_precision_many_2d.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(SINGLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

block_fft_2d_performance_many_double: block_fft_performance_many_2d.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(DOUBLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

block_fft_2d_performance_many_single: block_fft_performance_many_2d.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(SINGLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

# Targets for specific precision, dimension, and performance
.PHONY: precision performance
.PHONY: double single half
.PHONY: 1d 2d

precision_1d: $(TARGETS_1D_PRECISION)
precision_2d: $(TARGETS_2D_PRECISION)
performance_1d: $(TARGETS_1D_PERFORMANCE)
performance_2d: $(TARGETS_2D_PERFORMANCE)

double_precision_1d: block_fft_1d_precision_many_double
double_precision_2d: block_fft_2d_precision_many_double
double_performance_1d: block_fft_1d_performance_many_double
double_performance_2d: block_fft_2d_performance_many_double

single_precision_1d: block_fft_1d_precision_many_single
single_precision_2d: block_fft_2d_precision_many_single
single_performance_1d: block_fft_1d_performance_many_single
single_performance_2d: block_fft_2d_performance_many_single

half_precision_1d: block_fft_1d_precision_many_half
half_precision_2d: block_fft_2d_precision_many_half
half_performance_1d: block_fft_1d_performance_many_half
half_performance_2d: block_fft_2d_performance_many_half

# Precision and dimension combinations
double: double_precision_1d double_performance_1d double_precision_2d double_performance_2d
single: single_precision_1d single_performance_1d single_precision_2d single_performance_2d
half: half_precision_1d half_performance_1d half_precision_2d half_performance_2d

1d: precision_1d performance_1d
2d: precision_2d performance_2d

# Clean rule
.PHONY: clean
clean:
	rm -f $(TARGETS)
