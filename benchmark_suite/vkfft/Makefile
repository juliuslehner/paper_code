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

# Precision flags
PRECISION_FLAGS_single = -DSINGLE_PRECISION
PRECISION_FLAGS_double = -DDOUBLE_PRECISION
PRECISION_FLAGS_half = -DHALF_PRECISION

# Source files and corresponding output binaries
SRC1 = vkfft_performance_benchmark.cpp utils_VkFFT.cpp
OUT1 = vkfft_performance_benchmark
SRC2 = vkfft_precision_benchmark.cpp utils_VkFFT.cpp
OUT2 = vkfft_precision_benchmark

# Precision types
PRECISIONS = single double half

# Default target
.PHONY: all
all: $(OUT1) $(OUT2)

# Targets for the performance benchmark
.PHONY: $(OUT1)
$(OUT1): $(SRC1)
	$(NVCC) $(CFLAGS) $(BACKEND_FLAGS_cuda) $(SRC1) -o $@ $(PRECISION_FLAGS_$(PRECISION))

# Targets for the precision benchmark
.PHONY: $(OUT2)
$(OUT2): $(SRC2)
	$(NVCC) $(CFLAGS) $(BACKEND_FLAGS_cuda) $(SRC2) -o $@ $(PRECISION_FLAGS_$(PRECISION))

# Precision-specific builds
.PHONY: $(PRECISIONS)
$(PRECISIONS):
	@$(MAKE) $(OUT1) PRECISION=$@
	@$(MAKE) $(OUT2) PRECISION=$@

# Clean target to remove the output binaries
.PHONY: clean
clean:
	rm -f $(OUT1) $(OUT2)