#ifndef CUFFTDX_EXAMPLE_BLOCK_FFT_PERFORMANCE_HPP_
#define CUFFTDX_EXAMPLE_BLOCK_FFT_PERFORMANCE_HPP_

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <nvml.h>

#include "block_io.hpp"
#include "common.hpp"
#include "random.hpp"

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft_kernel(typename FFT::value_type*    data,
                                                                               unsigned int                 repeats,
                                                                               typename FFT::workspace_type workspace) {
    using complex_type = typename FFT::value_type;
    extern __shared__ unsigned char shared_mem[];

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load(data, thread_data, local_fft_id);

// Execute FFT
#pragma unroll 1
    for (unsigned int i = 0; i < repeats; i++) {
        FFT().execute(thread_data, shared_mem, workspace);
    }

    // Save results
    example::io<FFT>::store(thread_data, data, local_fft_id);
}

template<class FFTBase, unsigned int S/* Size */, unsigned int EPT, unsigned int FPB, bool UseSuggested = false>
void benchmark_block_fft(std::ofstream& file, const cudaStream_t& stream, double real_low, double real_high, double imag_low, double imag_high, int num_repeats, bool verbose = false) {
    using namespace cufftdx;

    // Create complete FFT description, only now we can query EPT and suggested FFTs per block
    using FFT_complete = decltype(FFTBase() + Size<S>());

    static constexpr unsigned int inside_repeats = 25000;
    static constexpr unsigned int kernel_runs = 5;
    static constexpr unsigned int warm_up_runs   = 1;

    static constexpr unsigned int fft_size            = S;
    static constexpr unsigned int elements_per_thread = UseSuggested ? FFT_complete::elements_per_thread : EPT;
    static constexpr unsigned int ffts_per_block      = UseSuggested ? FFT_complete::suggested_ffts_per_block : FPB;

    using FFT = decltype(FFT_complete() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>());
    using complex_type = typename FFT::value_type;

    double total_gflops = 0.0;
    double total_avg_duration = 0.0;
    double total_power = 0.0;
    double total_avg_energy = 0.0;
    for(int i = 0; i < num_repeats; i++){
        // Increase max shared memory if needed
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            block_fft_kernel<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size));

        int blocks_per_multiprocessor = 0;
        CUDA_CHECK_AND_EXIT(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                        block_fft_kernel<FFT>,
                                                        FFT::block_dim.x * FFT::block_dim.y * FFT::block_dim.z,
                                                        FFT::shared_memory_size));

        unsigned int multiprocessor_count = example::get_multiprocessor_count();
        unsigned int cuda_blocks = blocks_per_multiprocessor * multiprocessor_count;

        // The memory required to run fft (number of complex_type values that must be allocated).
        // For r2c, the input consists of fft_size real numbers and the output consists of (fft_size / 2 + 1) complex numbers.
        // One memory block will be used to store input and output, so the memory block must fit
        // max((fft_size + 1) / 2, fft_size / 2 + 1) = (fft_size / 2 + 1) complex numbers.
        // For c2r, the input consists of (fft_size / 2 + 1) complex numbers and the output consists of fft_size real numbers,
        // so the minimal required memory size is the same.
        unsigned int input_size =
            ffts_per_block * cuda_blocks * (type_of<FFT>::value == fft_type::c2c ? fft_size : (fft_size / 2 + 1));

        // Host data
        std::vector<complex_type> input =
    #ifdef HALF_PRECISION
            example::get_random_complex_data<__half>(input_size, __double2half(real_low), __double2half(real_high), __double2half(imag_low), __double2half(imag_high));
    #else
            example::get_random_complex_data<typename complex_type::value_type>(input_size, real_low, real_high, imag_low, imag_high);
    #endif
            

        // Device data
        complex_type* device_buffer;
        auto          size_bytes = input.size() * sizeof(complex_type);
        CUDA_CHECK_AND_EXIT(cudaMalloc(&device_buffer, size_bytes));
        // Copy host to device
        CUDA_CHECK_AND_EXIT(cudaMemcpy(device_buffer, input.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        cudaError_t error_code = cudaSuccess;
        auto        workspace  = make_workspace<FFT>(error_code);
        CUDA_CHECK_AND_EXIT(error_code);
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
        CUDA_CHECK_AND_EXIT(cudaGetLastError());

        // Measure performance of N trials
        auto ms_n_pair = example::measure_execution_ms(
            [&](cudaStream_t stream) {
                block_fft_kernel<FFT><<<cuda_blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                    device_buffer, inside_repeats, workspace);
            },
            warm_up_runs, kernel_runs, stream);

        // Check kernel error
        CUDA_CHECK_AND_EXIT(cudaGetLastError());

        // Copy host to device
        CUDA_CHECK_AND_EXIT(cudaMemcpy(device_buffer, input.data(), size_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        // Measure performance of 2*N trials
        auto ms_n2_pair = example::measure_execution_ms(
            [&](cudaStream_t stream) {
                block_fft_kernel<FFT><<<cuda_blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                    device_buffer, 2 * inside_repeats, workspace);
            },
            warm_up_runs, kernel_runs, stream);

        CUDA_CHECK_AND_EXIT(cudaFree(device_buffer));

        // Time for N repeats without overhead
        double   time_n = ms_n2_pair.first - ms_n_pair.first;
        double   energy_n = ms_n2_pair.second - ms_n_pair.second;
        unsigned int   total_ffts = inside_repeats * kernel_runs * ffts_per_block * cuda_blocks;
        // std::cout << "Total FFTs: " << total_ffts << std::endl;
        double power = energy_n/time_n;
        double avg_duration = (time_n/total_ffts)*1.0E-3;
        double avg_energy = energy_n/total_ffts;
        total_power += power;
        total_avg_duration += avg_duration;  
        total_avg_energy += avg_energy;
        double gflops = 5.0 * total_ffts * fft_size * log2(fft_size)/(time_n*1000000.0);
        // double gflops = 1.0 * kernel_runs * inside_repeats * ffts_per_block * cuda_blocks * 5.0 * fft_size *
            (std::log(fft_size) / std::log(2)) / time_n / 1000000.0;
        total_gflops += gflops;

        // static const std::string fft_type_name = type_of<FFT>::value == fft_type::c2c ? "c2c" :
                                                // (type_of<FFT>::value == fft_type::c2r ? "c2r" :
                                                // "r2c");
        if (verbose) {
            // std::cout << "FFT type: " << fft_type_name << std::endl;
            std::cout << "FFT size: " << fft_size << std::endl;
            std::cout << "FFTs elements per thread: " << FFT::elements_per_thread << std::endl;
            std::cout << "FFTs per block: " << ffts_per_block << std::endl;
            std::cout << "CUDA blocks: " << cuda_blocks << std::endl;
            std::cout << "Blocks per multiprocessor: " << blocks_per_multiprocessor << std::endl;
            std::cout << "FFTs run: " << ffts_per_block * cuda_blocks << std::endl;
            std::cout << "Shared memory: " << FFT::shared_memory_size << std::endl;
            std::cout << "Avg Time [ms_n]: " << time_n / (inside_repeats * kernel_runs) << std::endl;
            std::cout << "Time (all) [ms_n]: " << time_n << std::endl;
            std::cout << "Performance [GFLOPS]: " << gflops << std::endl;
            std::cout << "Avg Energy [mJ]: " << energy_n / (inside_repeats * kernel_runs) << std::endl;
            std::cout << "Total Energy [mJ]: " << energy_n << std::endl;
        } else{
            std::cout << "N = " << fft_size << ", Repeat " << i + 1 << ": Average Time: " << avg_duration << "s, GFLOPS: "
                      << gflops << ", Power: " << power << "W" << std::endl;
        }
    }
    // Average the accumulated values over the number of repeats
    double gflops_final = total_gflops / num_repeats;
    double avg_duration_final = total_avg_duration / num_repeats;
    double power_final = total_power / num_repeats;  
    double avg_energy_final = total_avg_energy / num_repeats;
    // std::cout << fft_type_name << ", " << fft_size << ", " << gflops << ", "
    //           << time_n / (inside_repeats * kernel_runs) << ", " << (double)power_n / (double)(inside_repeats * kernel_runs) << ", " << std::endl;
    file << real_low << "," << real_high << "," << imag_low << "," << imag_high << "," << fft_size << "," << num_repeats << "," << inside_repeats << "," << FFT::elements_per_thread << "," << ffts_per_block << "," << gflops_final << ","
                << avg_duration_final << "," << power_final << "," << avg_energy_final << "\n";
}

#endif // CUFFTDX_EXAMPLE_BLOCK_FFT_PERFORMANCE_HPP_
