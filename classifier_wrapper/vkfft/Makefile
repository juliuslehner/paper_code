# Define the backend (set it here)
BACKEND = cuda  # Change this to vulkan, hip, opencl, level_zero, or metal


# Define the compiler and flags
CUDA_TOOLKIT_ROOT_DIR := /usr/local/cuda
NVCC = nvcc
CXX = g++
CFLAGS = -I/usr/local/cuda/include -I/u/home/lehnj/benchmarking_suite/VkFFT/vkFFT -I/u/home/lehnj/benchmarking_suite/VkFFT/half_lib -I/u/home/lehnj/benchmarking_suite/VkFFT/benchmark_scripts/vkFFT_scripts/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcufft -lnvidia-ml -lfftw3l -lquadmath -lcuda -lnvrtc
VULKAN_FLAGS = -lvulkan -lspirv -lglslang

# Backend-specific flags
BACKEND_FLAGS_vulkan = -DVKFFT_BACKEND=0 $(VULKAN_FLAGS)
BACKEND_FLAGS_cuda = -DVKFFT_BACKEND=1 -DCUDA_TOOLKIT_ROOT_DIR=\"$(CUDA_TOOLKIT_ROOT_DIR)\" $(CFLAGS) $(LDFLAGS)
BACKEND_FLAGS_hip = -DVKFFT_BACKEND=2 -lhipfft
BACKEND_FLAGS_opencl = -DVKFFT_BACKEND=3 -lOpenCL
BACKEND_FLAGS_level_zero = -DVKFFT_BACKEND=4 -lze_loader
BACKEND_FLAGS_metal = -DVKFFT_BACKEND=5 -lFoundation -lQuartzCore -lMetal

# Debug each backend flag
$(info BACKEND_FLAGS_cuda=$(BACKEND_FLAGS_cuda))
$(info BACKEND_FLAGS_vulkan=$(BACKEND_FLAGS_vulkan))

# Dynamically assign BACKEND_FLAGS using eval
$(eval BACKEND_FLAGS := $(BACKEND_FLAGS_$(BACKEND)))

# Debugging info to confirm correct evaluation
$(info BACKEND_FLAGS_$(BACKEND)=$(BACKEND_FLAGS_$(BACKEND)))
$(info BACKEND=$(BACKEND))
$(info BACKEND_FLAGS=$(BACKEND_FLAGS))

# Precision flags
PRECISION_FLAGS_single = -DSINGLE_PRECISION
PRECISION_FLAGS_double = -DDOUBLE_PRECISION
PRECISION_FLAGS_half = -DHALF_PRECISION

# Source files and corresponding output binaries
SRC1D = vkfft_transform_1d.cpp utils_VkFFT.cpp
OUT1D = vkfft_transform_1d
SRC2D = vkfft_transform_2d.cpp utils_VkFFT.cpp
OUT2D = vkfft_transform_2d

# Precision types
PRECISIONS = single double half

# Default target
.PHONY: all
all: $(PRECISIONS)

# Precision targets (compile all dimensions)
.PHONY: $(PRECISIONS)
$(PRECISIONS): %: %_1d %_2d
	@echo "Built for precision: $@ with backend: $(BACKEND)"

# Dimension-specific targets for 1D
.PHONY: $(PRECISIONS:%=%_1d)
$(PRECISIONS:%=%_1d): %_1d: $(SRC1D)
	$(NVCC) $(CFLAGS) $(BACKEND_FLAGS_cuda) $(SRC1D) -o $(OUT1D)_$* $(PRECISION_FLAGS_$*)

# Dimension-specific targets for 2D
.PHONY: $(PRECISIONS:%=%_2d)
$(PRECISIONS:%=%_2d): %_2d: $(SRC2D)
	$(NVCC) $(CFLAGS) $(BACKEND_FLAGS_cuda) $(SRC2D) -o $(OUT2D)_$* $(PRECISION_FLAGS_$*)

# Clean target to remove the output binaries
.PHONY: clean
clean:
	rm -f $(OUT1D)_* $(OUT2D)_*
