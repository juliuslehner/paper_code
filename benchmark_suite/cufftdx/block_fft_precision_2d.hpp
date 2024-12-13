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
#include "fftw3.h"

// #define CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_2D
inline constexpr unsigned int cufftdx_example_warm_up_runs = 5;
inline constexpr unsigned int cufftdx_example_performance_runs = 1000;

struct Complex {
    double real;
    double imag;
};

std::vector<Complex> get_fftw_values(int Nx, int Ny, fftwl_complex* in){
    fftwl_complex* out = (fftwl_complex*) fftwl_malloc(sizeof(fftwl_complex) * Nx * Ny);
    fftwl_plan p = fftwl_plan_dft_2d(Nx, Ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwl_execute(p);
    // Read in output vector
    std::vector<Complex> data_highprecision(Nx*Ny);
    std::cout << "Expected Output: " << std::endl;
    for (size_t i = 0; i < (Nx*Ny); ++i) {
        data_highprecision[i].real = static_cast<double>(out[i][0]);
        data_highprecision[i].imag = static_cast<double>(out[i][1]);
        std::cout << data_highprecision[i].real << ", " << data_highprecision[i].imag << std::endl;
    }
    fftwl_destroy_plan(p);
    fftwl_free(in);
    fftwl_free(out);
    return data_highprecision;
}

void calculate_error(const std::vector<Complex>& data1, const std::vector<Complex>& data2, double *error_abs, double *error_rel) {
    using namespace std;
    if (data1.size() != data2.size()) {
        throw invalid_argument("Vectors must be of the same size.");
    }
    double sum_diff_abs = 0.0;
    double sum_diff_rel = 0.0;
	size_t num_entries = data1.size();
    double real_diff, imag_diff, abs_error, magnitude;
    for (size_t i = 0; i < data1.size(); ++i) {
        real_diff = data1[i].real - data2[i].real;
        imag_diff = data1[i].imag - data2[i].imag;
        abs_error = abs(real_diff) + abs(imag_diff);
        if (isnan(real_diff) || isnan(imag_diff) || isinf(real_diff) || isinf(imag_diff)) {
            // std::cerr << "Warning: Detected NaN or Inf value in data at index " << i << std::endl;
			// std::cout << "Warning: Detected NaN or Inf value in data at index " << i << std::endl;
			num_entries--;
            continue; // Skip this iteration
        }
        sum_diff_abs += abs_error;
        // Get relative errors
        magnitude = sqrt(data1[i].real * data1[i].real + data1[i].imag * data1[i].imag);
        if (magnitude != 0) {
            sum_diff_rel += abs_error / magnitude;
        } else {
            // Handle the case where magnitude is zero
            sum_diff_rel += abs_error; // or another appropriate measure
        }
    }
    if (num_entries > 0) {
        *error_rel = sum_diff_rel / num_entries;
        *error_abs = sum_diff_abs / num_entries;
    } else {
        *error_rel = -99.0;
        *error_abs = -99.0;
		cout << "No valid entries!!" << endl;
    }
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
std::vector<T> cufftdx_fft_2d(T* input, T* output, cudaStream_t stream) {
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
    static constexpr size_t flat_fft_size       = fft_size_x * fft_size_y;
    static constexpr size_t flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
    std::vector<complex_type> output_host(flat_fft_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Performance measurements
    auto time = example::measure_execution_ms(
        [&](cudaStream_t stream) {
            fft_2d_execution(stream);
        },
        cufftdx_example_warm_up_runs,
        cufftdx_example_performance_runs,
        stream).first;

    // Return results
    // return example::fft_results<T>{output_host};
    return output_host;
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
void precision_block_fft_2d(std::ofstream& error_file, double real_low, double real_high, double imag_low, double imag_high) {
    using namespace cufftdx;
    using fft_y    = decltype(fft_base() + Size<size_y>() + ElementsPerThread<ept_y>() + FFTsPerBlock<fpb_y>());
    using fft_x    = decltype(fft_base() + Size<size_x>() + ElementsPerThread<ept_x>() + FFTsPerBlock<fpb_x>());
    using fft      = fft_y;
    using complex_type = typename fft::value_type;

    // Host data
    static constexpr size_t flat_fft_size       = size_x * size_y;
    static constexpr size_t flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_2D
    std::vector<complex_type> input_host(flat_fft_size);
    for (size_t i = 0; i < flat_fft_size; i++) {
        float sign      = (i % 3 == 0) ? -1.0f : 1.0f;
        input_host[i].x = sign * static_cast<float>(i) / flat_fft_size;
        input_host[i].y = sign * static_cast<float>(i) / flat_fft_size;
    }
#else
    // std::cout << "Getting input signal" << std::endl;
    // Get input signal
    fftwl_complex *in = (fftwl_complex*) fftwl_malloc(sizeof(fftwl_complex) * (size_x * size_y));
    double real_mean = 0.0, imag_mean = 0.0, rate_change = 0.0 , magnitude_mean = 0.0;
    for(int i = 0; i < (flat_fft_size); i++){
        in[i][0] = (static_cast<long double>(rand()) / RAND_MAX) * (real_high - real_low) + real_low;
        in[i][1] = (static_cast<long double>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low;
        real_mean += in[i][0];
        imag_mean += in[i][1];
        if (i > 0) rate_change += std::abs(std::sqrt(in[i][0]*in[i][0] + in[i][1]*in[i][1]) - std::sqrt(in[i-1][0]*in[i-1][0] + in[i-1][1]*in[i-1][1]));
        magnitude_mean += std::sqrt(in[i][0]*in[i][0] + in[i][1]*in[i][1]);
    }
    rate_change /= (flat_fft_size-1);
    magnitude_mean /= flat_fft_size;
    real_mean /= flat_fft_size;
    imag_mean /= flat_fft_size;

    // Calculate Variance for imaginary and real part
    double variance_real = 0.0, variance_imag = 0.0;
    for(int i = 0; i < flat_fft_size; i++){
        variance_real += std::pow(in[i][0] - real_mean, 2);
        variance_imag += std::pow(in[i][1] - imag_mean, 2);
    }    
    variance_real /= flat_fft_size;
    variance_imag /= flat_fft_size;    
    double var_diff = std::abs(variance_imag - variance_real);
    // std::cout << "Input signal created: " << std::endl;
#ifdef HALF_PRECISION
    complex_type* input;
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&input, flat_fft_size_bytes));
#else
    std::vector<complex_type> input_host(flat_fft_size);
#endif

    for (size_t i = 0; i < (flat_fft_size); i++) {
#ifdef HALF_PRECISION
        float v1 = static_cast<float>(in[i][0]);
        float v2 = static_cast<float>(in[i][1]);
        // std::cout << in[i][0] << ", " << in[i][1] << "\n" << std::endl;
        // Populate input with complex<half2> values in ((Real, Imag), (Real, Imag)) layout
        input[i] = complex_type {__half2 {v1, v2}, __half2 {v1, v2}};
#elif defined(DOUBLE_PRECISION)
        input_host[i].x = static_cast<double>(in[i][0]);
        input_host[i].y = static_cast<double>(in[i][1]);
#else   
        input_host[i].x = static_cast<float>(in[i][0]);
        input_host[i].y = static_cast<float>(in[i][1]);
#endif
    }
    // auto input_host = example::get_random_complex_data<precision_type>(flat_fft_size, -1, 1);
#endif

    // std::cout << "Input Signal in host array" << std::endl; 
    // Device data
#ifndef HALF_PRECISION
    complex_type* input;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&input, flat_fft_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(input, input_host.data(), flat_fft_size_bytes, cudaMemcpyHostToDevice));
#endif
    complex_type* output;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, flat_fft_size_bytes));   
    // Copy host to device
    CUDA_CHECK_AND_EXIT(cudaMemset(output, 0, flat_fft_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // std::cout << "Good until cufftdx 2d call" << std::endl;
    // cuFFTDx 2D
    auto cufftdx_output = cufftdx_fft_2d<fft_x, fft_y, false>(input, output, stream);

    // cuFFTDx 2D
    // * Uses shared memory to speed-up IO in the strided kernel
    // auto cufftdx_smemio_results = cufftdx_fft_2d<fft_x, fft_y, true>(input, output, stream);


    // Destroy created CUDA stream
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    // Free CUDA buffers
    CUDA_CHECK_AND_EXIT(cudaFree(input));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    std::cout << "FFT: (" << size_x << ", " << size_y << ")\n";

#ifdef CUFFTDX_EXAMPLE_DETAIL_DEBUG_FFT_2D
    std::cout << "cuFFTDx\n";
    for (size_t i = 0; i < 8; i++) {
        std::cout << i << ": ";
        std::cout << "(" << cufftdx_output[i].x << ", " << cufftdx_output[i].y << ")";
        std::cout << "\n";
    }
#endif

    // Get the data of the fftw
    std::vector<Complex> data_double = get_fftw_values(size_x, size_y, in);

    std::cout << "Low Precision Data:\n" << std::endl;
    // Convert input_host to double precision for comparison
    std::vector<Complex> data_lowprecision(flat_fft_size);
    for (size_t i = 0; i < flat_fft_size; ++i) {
#ifdef HALF_PRECISION
        data_lowprecision[i].real = static_cast<double>(__half2float(cufftdx_output[i].x.x));
        data_lowprecision[i].imag = static_cast<double>(__half2float(cufftdx_output[i].x.y));
        std::cout << data_lowprecision[i].real << ", " << data_lowprecision[i].imag << "\n" << std::endl;
#else
        data_lowprecision[i].real = static_cast<double>(cufftdx_output[i].x);
        data_lowprecision[i].imag = static_cast<double>(cufftdx_output[i].y);
#endif
        // std::cout << data_lowprecision[i].real << " " << data_lowprecision[i].imag << std::endl;
    }
    
    // Calculate RMSE
    double abs_error, rel_error = 0.0;
    calculate_error(data_lowprecision, data_double, &abs_error, &rel_error);    
    error_file << variance_real << "," << variance_imag << "," << var_diff << "," << rate_change << "," << magnitude_mean << "," << size_x << "," << size_y << "," << abs_error << "," << rel_error*100 << "\n";      
//     bool success = true;
//     std::cout << "Correctness results:\n";
//     // Check if cuFFTDx results are correct
//     {
//         auto fft_error = example::fft_signal_error::calculate_for_complex_values(cufftdx_output, cufft_results.output);
//         std::cout << "cuFFTDx\n";
//         std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
//         std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";
//         if(success) {
//             success = (fft_error.l2_relative_error < 0.001);
//         }
//     }
//     // Check cuFFTDx with shared memory io
//     {
//         auto fft_error = example::fft_signal_error::calculate_for_complex_values(cufftdx_smemio_results.output, cufft_results.output);
//         std::cout << "cuFFTDx (shared memory IO)\n";
//         std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
//         std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";
//         if(success) {
//             success = (fft_error.l2_relative_error < 0.001);
//         }
//     }

//     // Print performance results
//     if(success) {
//         std::cout << "\nPerformance results:\n";
//         std::cout << std::setw(28) << "cuFFTDx: " << cufftdx_avg_time_in_ms << " [ms]\n";
//         std::cout << std::setw(28) << "cuFFTDx (shared memory IO): " << cufftdx_smemio_results.avg_time_in_ms << " [ms]\n";
//         std::cout << std::setw(28) << "cuFFT: " << cufft_results.avg_time_in_ms << " [ms]\n";
//     }

//     if (success) {
//         std::cout << "\nSuccess\n";
//     } else {
//         std::cout << "\nFailure\n";
//         std::exit(1);
//     }
}

// template<unsigned int Arch>
// struct fft_2d_functor {
//     void operator()() { return fft_2d<Arch>(); }
// };

// int main(int, char**) {
//     return example::sm_runner<fft_2d_functor>();
// }
