#ifndef CUFFTDX_TRANSFORM_1D_HPP_
#define CUFFTDX_TRANSFORM_1D_HPP_

#include <complex>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <iomanip>
#include <unistd.h>
// In cufftdx folder
#include <cufftdx.hpp>
#include <cuda_fp16.h>
#include "block_io.hpp"
#include "common.hpp"
#include "random.hpp"

// Create Struct to get complex data
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
#ifdef HALF_PRECISION
        float v1 = static_cast<float>(real);
        float v2 = static_cast<float>(imag);
        // Populate input with complex<half2> values in ((Real, Imag), (Real, Imag)) layout
        input[i] = ComplexType {__half2 {v1, v2}, __half2 {v1, v2}};
#elif defined(DOUBLE_PRECISION)
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
#ifdef HALF_PRECISION
        double real = static_cast<double>(__half2float(output[i].x.x));
        double imag = static_cast<double>(__half2float(output[i].x.y));
#elif defined(DOUBLE_PRECISION)
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

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load(data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io<FFT>::store(thread_data, data, local_fft_id);
}

template<class FFTBase, unsigned int S/* Size */, unsigned int EPT, unsigned int FPB, bool UseSuggested = false>
void block_fft_1d(std::string input_file, std::string output_file) {
    using namespace cufftdx;
    using FFT_complete = decltype(FFTBase() + Size<S>());

    //static constexpr unsigned int fft_size            = S;
    static constexpr unsigned int elements_per_thread = UseSuggested ? FFT_complete::elements_per_thread : EPT;
    static constexpr unsigned int ffts_per_block      = UseSuggested ? FFT_complete::suggested_ffts_per_block : FPB;
    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,
    using FFT = decltype(FFT_complete() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>());
#if CUFFTDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using complex_type = example::value_type_t<FFT>;
#else
    using complex_type = typename FFT::value_type;
#endif

    // Allocate managed memory for input/output
    complex_type* fft_data;
    constexpr size_t implicit_batching = FFT::implicit_type_batching;
    auto             size              = FFT::ffts_per_block / implicit_batching * cufftdx::size_of<FFT>::value;
    auto             size_bytes        = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&fft_data, size_bytes));    

    // Get input data from file
    read_binary(input_file, fft_data, cufftdx::size_of<FFT>::value);

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

    // Invokes kernel with FFT::block_dim threads in CUDA block
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(fft_data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Convert fft_data to double precision for comparison
    write_binary(output_file, fft_data, cufftdx::size_of<FFT>::value);

    CUDA_CHECK_AND_EXIT(cudaFree(fft_data));
}

#endif // CUFFTDX_EXAMPLE_BLOCK_FFT_PRECISION_HPP_