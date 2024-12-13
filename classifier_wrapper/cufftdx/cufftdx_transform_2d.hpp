#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <algorithm>
#include <complex>
#include <thrust/complex.h>
#include <sstream>
#include <fstream>
#include <cmath>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>

#include "block_io.hpp"
#include "block_io_strided.hpp"
#include "common.hpp"
#include "random.hpp"
#include "fp16_common.hpp"

struct Complex {
    double real;
    double imag;
};

template <typename ComplexType>
void read_binary(const std::string& filename, ComplexType* input, int num_elements) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file to read");
    }

    double real, imag;
    for (int i = 0; i < num_elements; ++i) {
        file.read(reinterpret_cast<char*>(&real), sizeof(double));
        file.read(reinterpret_cast<char*>(&imag), sizeof(double));
        if (!file) {
            throw std::runtime_error("Unexpected end of file while reading: " + filename);
        }
#ifdef DOUBLE_PRECISION
        input[i].x = static_cast<double>(real);
        input[i].y = static_cast<double>(imag);
#else   
        input[i].x = static_cast<float>(real);
        input[i].y = static_cast<float>(imag);
#endif
    }
    file.close();
}

template <typename ComplexType>
void write_binary(const std::string& filepath, ComplexType* output, int n) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file to write");
    }

    for (int i = 0; i < n; ++i) {
#ifdef DOUBLE_PRECISION
        double real = output[i].x;
        double imag = output[i].y;
#else
        double real = static_cast<double>(output[i].x);
        double imag = static_cast<double>(output[i].y);
#endif
        file.write(reinterpret_cast<const char*>(&real), sizeof(double));
        file.write(reinterpret_cast<const char*>(&imag), sizeof(double));
    }
        
    file.close();
}

template<class FFT, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void fft_2d_kernel_y(const ComplexType* input, ComplexType* output, typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load(input, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    example::io<FFT>::store(thread_data, output, local_fft_id);
}

template<class FFT, unsigned int Stride, bool UseSharedMemoryStridedIO, class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__
    void fft_2d_kernel_x(const ComplexType* input, ComplexType* output, typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;

    extern __shared__ complex_type shared_mem[];

    // Local array for thread
    ComplexType thread_data[FFT::storage_size];
    
    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    if constexpr (UseSharedMemoryStridedIO) {
        example::io_strided<FFT>::load_strided<Stride>(input, thread_data, shared_mem, local_fft_id);
    } else {
        example::io_strided<FFT>::load_strided<Stride>(input, thread_data, local_fft_id);
    }

    // Execute FFT
    FFT().execute(thread_data, shared_mem, workspace);

    // Save results
    if constexpr (UseSharedMemoryStridedIO) {
        example::io_strided<FFT>::store_strided<Stride>(thread_data, shared_mem, output, local_fft_id);
    } else {
        example::io_strided<FFT>::store_strided<Stride>(thread_data, output, local_fft_id);
    }
}

template<class FFTX, class FFTY, bool UseSharedMemoryStridedIO, class T>
void cufftdx_fft_2d(T* input, T* output, cudaStream_t stream) {
    using complex_type                       = typename FFTX::value_type;
    static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
    static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;

    // Checks that FFTX and FFTY are correctly defined
    static_assert(std::is_same_v<cufftdx::precision_of_t<FFTX>, cufftdx::precision_of_t<FFTY>>,
                  "FFTY and FFTX must have the same precision");
    static_assert(std::is_same_v<typename FFTX::value_type, typename FFTY::value_type>,
                  "FFTY and FFTX must operator on the same type");
    static_assert(sizeof(T) == sizeof(complex_type), "");
    static_assert(std::alignment_of_v<T> == std::alignment_of_v<complex_type>, "");
    // Checks below are not caused by any limitation in cuFFTDx, but rather in the example IO functions.
    static_assert((fft_size_x % FFTY::ffts_per_block == 0),
                  "FFTsPerBlock for FFTX must divide Y dimension as IO doesn't check if a batch is in range");

    // complex_type* cufftdx_input  = reinterpret_cast<complex_type*>(input);
    // complex_type* cufftdx_output = reinterpret_cast<complex_type*>(output);

    // Set shared memory requirements
    auto error_code = cudaFuncSetAttribute(
        fft_2d_kernel_y<FFTY, complex_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFTY::shared_memory_size);
    CUDA_CHECK_AND_EXIT(error_code);

    // Shared memory IO for strided kernel may require more memory than FFTX::shared_memory_size.
    // Note: For some fft_size_x and depending on GPU architecture fft_x_shared_memory_smem_io may exceed max shared
    // memory and cudaFuncSetAttribute will fail.
    unsigned int fft_x_shared_memory_smem_io =
        std::max<unsigned>({FFTX::shared_memory_size, FFTX::ffts_per_block * fft_size_x * sizeof(complex_type)});
    unsigned int fft_x_shared_memory =
        UseSharedMemoryStridedIO ? fft_x_shared_memory_smem_io : FFTX::shared_memory_size;
    error_code = cudaFuncSetAttribute(fft_2d_kernel_x<FFTX, fft_size_y, UseSharedMemoryStridedIO, complex_type>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      fft_x_shared_memory);
    CUDA_CHECK_AND_EXIT(error_code);

    // Create workspaces for FFTs
    auto workspace_y = cufftdx::make_workspace<FFTY>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_x = cufftdx::make_workspace<FFTX>(error_code);
    CUDA_CHECK_AND_EXIT(error_code);

    // Synchronize device before execution
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Define 2D FFT execution
    const auto grid_fft_size_y = ((fft_size_x + FFTY::ffts_per_block - 1) / FFTY::ffts_per_block);
    const auto grid_fft_size_x = ((fft_size_y + FFTX::ffts_per_block - 1) / FFTX::ffts_per_block);
    auto fft_2d_execution = [&](cudaStream_t stream) {
        fft_2d_kernel_y<FFTY, complex_type><<<grid_fft_size_y, FFTY::block_dim, FFTY::shared_memory_size, stream>>>(
            input, output, workspace_y);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
        fft_2d_kernel_x<FFTX, fft_size_y, UseSharedMemoryStridedIO, complex_type>
            <<<grid_fft_size_x, FFTX::block_dim, fft_x_shared_memory, stream>>>(
                output, output, workspace_x);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
    };

    // Correctness run
    fft_2d_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // Copy results to host
    // static constexpr size_t flat_fft_size       = fft_size_x * fft_size_y;
    // static constexpr size_t flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
    // std::vector<complex_type> output_host(flat_fft_size);
    // CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    // CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // // Return results
    // return output_host;
}

// Example showing how cuFFTDx can be used to perform a 2D FFT in 2 kernels. The first kernel performs FFT along the contiguous
// dimension (Y), and the 2nd along the strided one.
//
// Notes:
// * This examples shows how to use cuFFTDx to run multi-dimensional FFT. Final performance will vary depending on the
// FFT definitions (precision, size, type, ept, fpb) and other user customizations.
// * Best possible performance requires adapting parameters in the sample to particular set of parameters and code customizations.
// * Only FP32 was tested for this example, other types might require adjustments.
// * cuFFTDx with enabled shared memory IO usually be the faster cuFFTDx option for larger (>512) sizes.
// * The shared memory IO cuFFTDx has high shared memory requirements and will not work for all possible sizes in X dimension.
template<class fft_base, unsigned int size_x, unsigned int size_y, unsigned int ept_x, unsigned int fpb_x, unsigned int ept_y, unsigned int fpb_y, bool UseSuggested = false>
void block_fft_2d(std::string input_file, std::string output_file) {
    using namespace cufftdx;
    using fft_y    = decltype(fft_base() + Size<size_y>() + ElementsPerThread<ept_y>() + FFTsPerBlock<fpb_y>());
    using fft_x    = decltype(fft_base() + Size<size_x>() + ElementsPerThread<ept_x>() + FFTsPerBlock<fpb_x>());
    using fft      = fft_y;
    using complex_type = typename fft::value_type;

    // Host data
    static constexpr size_t flat_fft_size       = size_x * size_y;
    static constexpr size_t flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);

    complex_type* input;
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&input, flat_fft_size_bytes));

    complex_type* output;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, flat_fft_size_bytes)); 

    // Get input data from file
    read_binary(input_file, input, flat_fft_size);

    // Copy host to device
    CUDA_CHECK_AND_EXIT(cudaMemset(output, 0, flat_fft_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // std::cout << "Good until cufftdx 2d call" << std::endl;
    // cuFFTDx 2D
    cufftdx_fft_2d<fft_x, fft_y, false>(input, output, stream);
    write_binary(output_file, output, flat_fft_size);
    // Destroy created CUDA stream
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    // Free CUDA buffers
    CUDA_CHECK_AND_EXIT(cudaFree(input));
    CUDA_CHECK_AND_EXIT(cudaFree(output));     
}
