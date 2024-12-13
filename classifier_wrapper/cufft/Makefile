# Define the compiler and flags
NVCC = nvcc
CFLAGS = -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcufft -lnvidia-ml -lfftw3l

# Source files and corresponding output binaries
SRC1 = cufft_transform_1d.cpp
OUT1 = cufft_transform_1d

SRC2 = cufft_transform_2d.cpp
OUT2 = cufft_transform_2d

# Precision types
PRECISIONS = single double half

# Default target
.PHONY: all
all: $(PRECISIONS)

# Combined targets for different precision types
.PHONY: $(PRECISIONS)
$(PRECISIONS): %: %_transform_1d %_transform_2d

# Targets for different precision types for the first benchmark
.PHONY: $(addsuffix _transform_1d,$(PRECISIONS))
$(addsuffix _transform_1d,$(PRECISIONS)): %_transform_1d: $(SRC1)
	$(NVCC) $(CFLAGS) $(LDFLAGS) $(SRC1) -o $(OUT1)_$* -D$(shell echo $* | tr a-z A-Z)_PRECISION

# Targets for different precision types for the second benchmark
.PHONY: $(addsuffix _transform_2d,$(PRECISIONS))
$(addsuffix _transform_2d,$(PRECISIONS)): %_transform_2d: $(SRC2)
	$(NVCC) $(CFLAGS) $(LDFLAGS) $(SRC2) -o $(OUT2)_$* -D$(shell echo $* | tr a-z A-Z)_PRECISION

# Clean target to remove the output binaries
.PHONY: clean
clean:
	rm -f $(OUT1)_* $(OUT2)_*
