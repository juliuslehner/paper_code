#ifndef CUFFTDX_EXAMPLE_BLOCK_FFT_PRECISION_HPP_
#define CUFFTDX_EXAMPLE_BLOCK_FFT_PRECISION_HPP_

#include <iostream>
#include <vector>
#include <thrust/complex.h>
#include <sstream>
#include <fstream>
#include <cmath>
#include "fftw3.h"

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "block_io.hpp"
#include "common.hpp"

// Create Struct to get complex data
struct Complex {
    double real;
    double imag;
};

std::vector<Complex> get_fftw_values(int N, fftwl_complex* in){
    fftwl_complex* out = (fftwl_complex*) fftwl_malloc(sizeof(fftwl_complex) * N);
    fftwl_plan p = fftwl_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwl_execute(p);
    // Read in output vector
    std::vector<Complex> data_highprecision(N);
    for (size_t i = 0; i < N; ++i) {
        data_highprecision[i].real = static_cast<double>(out[i][0]);
        data_highprecision[i].imag = static_cast<double>(out[i][1]);
    }
    fftwl_destroy_plan(p);
    fftwl_free(in);
    fftwl_free(out);
    return data_highprecision;
}

// Read in the CSV file
std::vector<Complex> read_csv(const std::string& filename) {
    std::vector<Complex> data;
    std::ifstream file(filename);
    std::string line;

    // Skip the header
    std::getline(file, line);

    // Read data
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        Complex c;

        // Skip first two columns
        std::getline(iss, token, ',');
        std::getline(iss, token, ',');

        std::getline(iss, token, ',');
        c.real = std::stod(token);
        std::getline(iss, token, ',');
        c.imag = std::stod(token);
        data.push_back(c);
    }

    return data;
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

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block.
//
// One block is run, it calculates two 128-point C2C float precision FFTs.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
template<class FFTBase, unsigned int S/* Size */, unsigned int EPT, unsigned int FPB, bool UseSuggested = false>
void precision_block_fft(std::ofstream& error_file, double real_low, double real_high, double imag_low, double imag_high) {
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
    // auto          size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    // auto          size_bytes = size * sizeof(complex_type);
    // CUDA_CHECK_AND_EXIT(cudaMallocManaged(&fft_data, size_bytes));

    constexpr size_t implicit_batching = FFT::implicit_type_batching;
    auto             size              = FFT::ffts_per_block / implicit_batching * cufftdx::size_of<FFT>::value;
    auto             size_bytes        = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&fft_data, size_bytes));    

    // Get input data from file
    // std::vector<Complex> input_data = read_csv(input_file);
    // Get input signal
    fftwl_complex *in = (fftwl_complex*) fftwl_malloc(sizeof(fftwl_complex) * size);
    double real_mean = 0.0, imag_mean = 0.0, rate_change = 0.0 , magnitude_mean = 0.0;
    for(int i = 0; i < size; i++){
        in[i][0] = (static_cast<long double>(rand()) / RAND_MAX) * (real_high - real_low) + real_low;
        in[i][1] = (static_cast<long double>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low;
        real_mean += in[i][0];
        imag_mean += in[i][1];
        if (i > 0) rate_change += std::abs(std::sqrt(in[i][0]*in[i][0] + in[i][1]*in[i][1]) - std::sqrt(in[i-1][0]*in[i-1][0] + in[i-1][1]*in[i-1][1]));
        magnitude_mean += std::sqrt(in[i][0]*in[i][0] + in[i][1]*in[i][1]);
    }
    rate_change /= (size-1);
    magnitude_mean /= size;
    real_mean /= size;
    imag_mean /= size;

    // Calculate Variance for imaginary and real part
    double variance_real = 0.0, variance_imag = 0.0;
    for(int i = 0; i < size; i++){
        variance_real += std::pow(in[i][0] - real_mean, 2);
        variance_imag += std::pow(in[i][1] - imag_mean, 2);
    }    
    variance_real /= size;
    variance_imag /= size;    
    double var_diff = std::abs(variance_imag - variance_real);

    // Populate input array according to accuracy
    for (size_t i = 0; i < size; i++) {
#ifdef HALF_PRECISION
        float v1 = static_cast<float>(in[i][0]);
        float v2 = static_cast<float>(in[i][1]);
        // Populate input with complex<half2> values in ((Real, Imag), (Real, Imag)) layout
        fft_data[i] = complex_type {__half2 {v1, v2}, __half2 {v1, v2}};
#elif defined(DOUBLE_PRECISION)
        fft_data[i].x = static_cast<double>(in[i][0]);
        fft_data[i].y = static_cast<double>(in[i][1]);
#else   
        fft_data[i].x = static_cast<float>(in[i][0]);
        fft_data[i].y = static_cast<float>(in[i][1]);
#endif
        
    }


    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

    // Invokes kernel with FFT::block_dim threads in CUDA block
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(fft_data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Get the data of the fftw
    std::vector<Complex> data_double = get_fftw_values(size, in);

    // Convert fft_data to double precision for comparison
    std::vector<Complex> data_lowprecision(cufftdx::size_of<FFT>::value);
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; ++i) {
#ifdef HALF_PRECISION
        data_lowprecision[i].real = static_cast<double>(__half2float(fft_data[i].x.x));
        data_lowprecision[i].imag = static_cast<double>(__half2float(fft_data[i].x.y));
#else
        data_lowprecision[i].real = static_cast<double>(fft_data[i].x);
        data_lowprecision[i].imag = static_cast<double>(fft_data[i].y);
#endif
        // std::cout << data_lowprecision[i].real << " " << data_lowprecision[i].imag << std::endl;
    }

    // Calculate RMSE
    double abs_error, rel_error = 0.0;
    calculate_error(data_lowprecision, data_double, &abs_error, &rel_error);
    // static const std::string fft_type_name = type_of<FFT>::value == fft_type::c2c ? "c2c" :
    //                                          (type_of<FFT>::value == fft_type::c2r ? "c2r" :
    //                                          "r2c");    
    std::cout << "N: " << size << ", Average Relative Error: " << rel_error*100 << "%" << std::endl;
    //error_file << size << ", " << FFT::elements_per_thread << ", " << ffts_per_block << ", " << abs_error << "," << rel_error*100.0 << "\n";  
    error_file << variance_real << "," << variance_imag << "," << var_diff << "," << rate_change << "," << magnitude_mean << "," << size << "," << abs_error << "," << rel_error*100 << "\n";  
    //std::cout << "Average Error between double precision cuFFTDx and double precision FFTW: " << rmse << std::endl;
    // //Construct filename
    // std::ostringstream filename;
    // filename << "output_size" << size
    //          << "_precision" << (std::is_same<precision, float>::value ? "float" : (std::is_same<precision, double>::value ? "double" : "half"))
    //          << "_ept" << ept
    //          << "_fpb" << fpb << ".csv";

    // // Open file in write mode
    // std::ofstream file(filename.str());

    // // Set the precision of the output
    // //file << std::fixed << std::setprecision(7);

    // // Write header
    // file << "RealX,ImagX\n";

    // // Save output to file
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++){
    //     file << fft_data[i].x << "," << fft_data[i].y << "\n";
    // }

    /* std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    } */

    // Close file
    // file.close();
    CUDA_CHECK_AND_EXIT(cudaFree(fft_data));
}

#endif // CUFFTDX_EXAMPLE_BLOCK_FFT_PRECISION_HPP_
