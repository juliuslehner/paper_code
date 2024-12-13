#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <string>
#include <iomanip>
#include <unistd.h>

#include "block_fft_performance.hpp"

extern char *optarg;
extern int optopt;

std::string double_to_string(double value, int precision = 1){
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    std::string result = ss.str();
    std::replace(result.begin(), result.end(), '.', '_');
    return result;
}

template<unsigned int      Arch,
         unsigned int      FFTSize,
         cufftdx::fft_type FFTType,
         class PrecisionType,
         cufftdx::fft_direction FFTDirection      = cufftdx::fft_direction::forward,
         bool                   UseSuggested      = true,
         unsigned int           ElementsPerThread = 2,
         unsigned int           FFTsPerBlock      = 1>
void block_fft_performance(const cudaStream_t& stream, bool verbose, std::ofstream& file,
                           double real_low, double real_high, double imag_low, double imag_high, int num_repeats) {
    using namespace cufftdx;

    using FFT_base = decltype(Block() + Type<FFTType>() + Precision<PrecisionType>() + SM<Arch>());

    using FFT_with_direction = typename std::
        conditional<FFTType == fft_type::c2c, decltype(FFT_base() + Direction<FFTDirection>()), FFT_base>::type;

    benchmark_block_fft<FFT_with_direction, FFTSize, ElementsPerThread, FFTsPerBlock, UseSuggested>(file, stream,real_low, real_high, imag_low, imag_high, num_repeats, verbose);

    std::cout << "done\n";
    // if (verbose)
    //     std::cout << std::endl;
}

template<unsigned int Arch = 800>
struct block_fft_performance_functor {
    double real_low, real_high, imag_low, imag_high;
    int rep_number;
    std::string output_file;

    block_fft_performance_functor(double rl, double rh, double il, double ih, int rn, std::string file)
        : real_low(rl), real_high(rh), imag_low(il), imag_high(ih), rep_number(rn), output_file(file) {}

    void operator()() {
        using namespace cufftdx;

        cudaStream_t stream;
        CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

        bool default_verbose = false;
        std::string precision;
#ifdef SINGLE_PRECISION
        precision = "single";
#elif defined(DOUBLE_PRECISION)
        precision = "double";
#else 
        precision = "half";
#endif
        // Construct filename depending on input
        std::stringstream filename;
        filename << output_file
                << precision << "_prec_benchmark_imag" << double_to_string(imag_low,4) << "to" << double_to_string(imag_high,4) << "_real" << double_to_string(real_low,4) << "to" << double_to_string(real_high,4) <<  ".csv";
        std::string filename_str = filename.str();

        // File to save the results
        std::ofstream file(filename_str);

        // Write header
        file << "re_low,re_high,im_low,im_high,N,Num_Repeats,Num_Runs,EPT,FPB,GFLOPS,AverageTime(s),Power(W),AverageEnergy(mJ)\n";
        int num_repeats = 1;
        // To specify EPT and FPB values, set UsedSuggested to false.
        // FFTDirection is used if and only if FFTType is C2C.
        // Below is an example of a test run with specified EPT and FPB values.

        // block_fft_performance<Arch, 137, fft_type::c2c, __half, fft_direction::forward, false, 2, 1>(stream,
        //           
#ifdef SINGLE_PRECISION
        block_fft_performance<Arch, 2, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 4, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 8, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 16, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 32, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 64, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 128, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 256, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 512, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 1024, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 2048, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 4096, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 8192, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 16384, fft_type::c2c, float>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
#elif defined(DOUBLE_PRECISION)                                                                              
        block_fft_performance<Arch, 2, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 4, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 8, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 16, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 32, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 64, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 128, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 256, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 512, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 1024, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 2048, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 4096, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 8192, fft_type::c2c, double>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
#else
        block_fft_performance<Arch, 2, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 4, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 8, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 16, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 32, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 64, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 128, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 256, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 512, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 1024, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 2048, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 4096, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 8192, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);
        block_fft_performance<Arch, 16384, fft_type::c2c, __half>(stream, default_verbose, file, real_low, real_high, imag_low, imag_high, num_repeats);

#endif
        file.close();
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    }
};

int main(int argc, char** argv) {
    double imag_low = -10.0, imag_high = 10.0;
    double real_low = -10.0, real_high = 10.0;
    int rep_number = 1;
    std::string output_filepath;

    // Get Arguments
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "1:2:3:4:b:o:"))) {
        switch (opt_c) {
        case '1':
            real_low = atof(optarg);
            break;
        case '2':
            real_high = atof(optarg);
            break;
        case '3':
            imag_low = atof(optarg);
            break;
        case '4':
            imag_high = atof(optarg);
            break;
        case 'b':
            rep_number = atoi(optarg);
            break;
        case 'o':
            output_filepath = strdup(optarg);
            break;
        case '?':
            std::cerr << "Unknown option: " << optopt << std::endl;
            break;
        default:
            break;
        }
    }

    // Instantiate the functor with the parsed parameters
    block_fft_performance_functor<800> functor(real_low, real_high, imag_low, imag_high, rep_number, output_filepath);
    // return example::sm_runner(functor);
    functor();
}
