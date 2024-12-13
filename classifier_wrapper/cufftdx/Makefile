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
SRCS1 := cufftdx_transform_1d.cu
SRCS2 := cufftdx_transform_2d.cu

TARGETS1 := cufftdx_transform_1d_half cufftdx_transform_1d_single cufftdx_transform_1d_double
TARGETS2 := cufftdx_transform_2d_single cufftdx_transform_2d_double
TARGETS := $(TARGETS1) $(TARGETS2)

# Precision-specific compiler flags
DOUBLE_FLAGS := -DDOUBLE_PRECISION
SINGLE_FLAGS := -DSINGLE_PRECISION
HALF_FLAGS := -DHALF_PRECISION

# Default target
all: $(TARGETS)

# Rules for 1D FFT different precisions
cufftdx_transform_1d_half: cufftdx_transform_1d.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(HALF_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

cufftdx_transform_1d_single: cufftdx_transform_1d.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(SINGLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

cufftdx_transform_1d_double: cufftdx_transform_1d.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(DOUBLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

# Rules for 2D FFT different precisions
cufftdx_transform_2d_single: cufftdx_transform_2d.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(SINGLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

cufftdx_transform_2d_double: cufftdx_transform_2d.cu
	$(NVCC) $(COMMON_CXXFLAGS) $(DOUBLE_FLAGS) $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS) $< -o $@

# Combined precision and dimension targets
.PHONY: double_1d double_2d single_1d single_2d half_1d

double_1d: cufftdx_transform_1d_double
double_2d: cufftdx_transform_2d_double
single_1d: cufftdx_transform_1d_single
single_2d: cufftdx_transform_2d_single
half_1d: cufftdx_transform_1d_half

# Specific targets
.PHONY: double single half
.PHONY: 1d 2d

double: double_1d double_2d
single: single_1d single_2d
half: half_1d

# Dimension-specific combined targets
1d: $(TARGETS1)
2d: $(TARGETS2)

# Clean rule
.PHONY: clean
clean:
	rm -f $(TARGETS)
