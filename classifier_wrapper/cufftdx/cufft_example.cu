#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "block_io.hpp"
#include "common.hpp"
#include "random.hpp"

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
template<unsigned int Arch>
void cufftdx_transform_1d() {
    using namespace cufftdx;
    static constexpr unsigned int ept = 8;
    static constexpr unsigned int fft_size = 128;
    static constexpr unsigned int fpb = 1;
    using precision = double;
    // FFT is defined, its: size, type, direction, precision. Block() operator informs that FFT
    // will be executed on block level. Shared memory is required for co-operation between threads.
    // Additionally,
    using FFT          = decltype(Block() + Size<fft_size>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<precision>() + ElementsPerThread<ept>() + FFTsPerBlock<fpb>() + SM<Arch>());
    #if CUFFTDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using complex_type = example::value_type_t<FFT>;
    #else
    using complex_type = typename FFT::value_type;
    #endif

    // Allocate managed memory for input/output
    complex_type* data;
    auto          size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto          size_bytes = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, size_bytes));

    // Input data generation
    std::vector<complex_type> input =
            example::get_random_complex_data<typename complex_type::value_type>(size, -10, 10);

    for (size_t i = 0; i < size; i++) {
        data[i].x = input[i].x;
        data[i].y = input[i].y;
    }

    // // Device data
    // complex_type* device_buffer;
    // auto          size_bytes = input.size() * sizeof(complex_type);
    // CUDA_CHECK_AND_EXIT(cudaMalloc(&device_buffer, size_bytes));    
    
    // Construct input filename
    std::ostringstream inputfilename;
    inputfilename << "input_size" << fft_size
             << "_precision" << (std::is_same<precision, float>::value ? "float" : (std::is_same<precision, double>::value ? "double" : "half"))
             << "_rand_-10to10" << ".csv";

    // Open file in write mode
    std::ofstream inputfile(inputfilename.str());
    inputfile << "RealX,ImagX\n";

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
        inputfile << data[i].x << "," << data[i].y << "\n";
    }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

    // Invokes kernel with FFT::block_dim threads in CUDA block
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    //Construct filename
    std::ostringstream filename;
    filename << "output_size" << fft_size
             << "_precision" << (std::is_same<precision, float>::value ? "float" : (std::is_same<precision, double>::value ? "double" : "half"))
             << "_ept" << ept
             << "_fpb" << fpb << ".csv";

    // Open file in write mode
    std::ofstream file(filename.str());

    // Set the precision of the output
    //file << std::fixed << std::setprecision(7);

    // Write header
    file << "RealX,ImagX\n";

    // Save output to file
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++){
        file << data[i].x << "," << data[i].y << "\n";
    }

    /* std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    } */

    // Close file
    file.close();
    CUDA_CHECK_AND_EXIT(cudaFree(data));
    std::cout << "Success" << std::endl;
}

template<unsigned int Arch>
struct cufftdx_transform_1d_functor {
    void operator()() { return cufftdx_transform_1d<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<cufftdx_transform_1d_functor>();
}
