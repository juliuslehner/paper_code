# Define the compiler and flags
NVCC = nvcc
CFLAGS = -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcufft -lnvidia-ml -lfftw3l

# Source files and corresponding output binaries
SRC1 = cufft_precision_benchmark_1d.cpp
OUT1 = cufft_precision_benchmark_1d

SRC2 = cufft_performance_benchmark_1d.cpp
OUT2 = cufft_performance_benchmark_1d 

SRC3 = cufft_batch_performance.cpp
OUT3 = cufft_batch_performance

SRC4 = cufft_precision_benchmark_2d.cpp
OUT4 = cufft_precision_benchmark_2d

SRC5 = cufft_performance_benchmark_2d.cpp
OUT5 = cufft_performance_benchmark_2d

# Precision types
PRECISIONS = single double half

# Default target
.PHONY: all
all: $(PRECISIONS)

# Combined targets for different precision types
.PHONY: $(PRECISIONS)
$(PRECISIONS): %: %_precision_1d %_performance_1d %_precision_2d %_performance_2d

# Helper function to generate the uppercase precision flag
uppercase = $(shell echo $1 | tr a-z A-Z)

# Targets for different precision types for the first benchmark
.PHONY: $(addsuffix _precision_1d,$(PRECISIONS))
$(addsuffix _precision_1d,$(PRECISIONS)): %_precision_1d: $(SRC1)
	$(NVCC) $(CFLAGS) $(LDFLAGS) $< -o $(OUT1) -D$(call uppercase,$*)_PRECISION

# Targets for different precision types for the second benchmark
.PHONY: $(addsuffix _performance_1d,$(PRECISIONS))
$(addsuffix _performance_1d,$(PRECISIONS)): %_performance_1d: $(SRC2)
	$(NVCC) $(CFLAGS) $(LDFLAGS) $< -o $(OUT2) -D$(call uppercase,$*)_PRECISION

# Targets for different precision types for the third benchmark (batch performance)
.PHONY: $(addsuffix _batch_performance,$(PRECISIONS))
$(addsuffix _batch_performance,$(PRECISIONS)): %_batch_performance: $(SRC3)
	$(NVCC) $(CFLAGS) $(LDFLAGS) $< -o $(OUT3) -D$(call uppercase,$*)_PRECISION

# Targets for different precision types for the 2D precision benchmark
.PHONY: $(addsuffix _precision_2d,$(PRECISIONS))
$(addsuffix _precision_2d,$(PRECISIONS)): %_precision_2d: $(SRC4)
	$(NVCC) $(CFLAGS) $(LDFLAGS) $< -o $(OUT4) -D$(call uppercase,$*)_PRECISION

# Targets for different precision types for the 2D performance benchmark
.PHONY: $(addsuffix _performance_2d,$(PRECISIONS))
$(addsuffix _performance_2d,$(PRECISIONS)): %_performance_2d: $(SRC5)
	$(NVCC) $(CFLAGS) $(LDFLAGS) $< -o $(OUT5) -D$(call uppercase,$*)_PRECISION

# Clean target to remove the output binaries
.PHONY: clean
clean:
	rm -f $(OUT1) $(OUT2) $(OUT3) $(OUT4) $(OUT5)
