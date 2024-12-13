#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <string>
#include <iomanip>
#include <unistd.h>

extern char *optarg;
extern int optopt;

#include "block_fft_precision.hpp"

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
void block_fft_precision(std::ofstream& error_file, double real_low, double real_high, double imag_low, double imag_high) {
    using namespace cufftdx;

    using FFT_base = decltype(Block() + Type<FFTType>() + Precision<PrecisionType>() + SM<Arch>());

    using FFT_with_direction = typename std::
        conditional<FFTType == fft_type::c2c, decltype(FFT_base() + Direction<FFTDirection>()), FFT_base>::type;

    precision_block_fft<FFT_with_direction, FFTSize, ElementsPerThread, FFTsPerBlock, UseSuggested>(error_file, real_low, real_high, imag_low, imag_high);

    std::cout << "done\n";
    // if (verbose)
    //     std::cout << std::endl;
}

template<unsigned int Arch = 800>
struct block_fft_precision_functor {
    double real_low, real_high, imag_low, imag_high;
    std::string output_file;

    block_fft_precision_functor(double rl, double rh, double il, double ih, std::string file)
        : real_low(rl), real_high(rh), imag_low(il), imag_high(ih), output_file(file) {}

    void operator()() {
        using namespace cufftdx;
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
                 << precision << "_prec_benchmark_imag" << double_to_string(imag_low,4) << "to" << double_to_string(imag_high,4) << "_real" << double_to_string(real_low,4) << "to" << double_to_string(real_high,4) << ".csv";
        std::string filename_str = filename.str();

        // File to save the results
        std::ofstream error_file(filename_str);

        // Write header
        error_file << "variance_real,variance_imag,variance_diff,rate_of_change,Magnitude,N,Absolute_Error,Relative_Error[%]\n";

        // To specify EPT and FPB values, set UsedSuggested to false.
        // FFTDirection is used if and only if FFTType is C2C.
#ifdef DOUBLE_PRECISION
        block_fft_precision<Arch, 2, fft_type::c2c, double, fft_direction::forward, false, 2, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 4, fft_type::c2c, double, fft_direction::forward, false, 4, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 8, fft_type::c2c, double, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 16, fft_type::c2c, double, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 32, fft_type::c2c, double, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 64, fft_type::c2c, double, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 128, fft_type::c2c, double, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 256, fft_type::c2c, double, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 512, fft_type::c2c, double, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 1024, fft_type::c2c, double, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 2048, fft_type::c2c, double, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 4096, fft_type::c2c, double, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 8192, fft_type::c2c, double, fft_direction::forward, false, 16, 1>(error_file, real_low, real_high, imag_low, imag_high);
        // block_fft_precision<Arch, 16384, fft_type::c2c, double, fft_direction::forward, false, 32, 1>(inputFilenames[13], outputFilenames[13], error_file);
#elif defined(SINGLE_PRECISION)
        block_fft_precision<Arch, 2, fft_type::c2c, float, fft_direction::forward, false, 2, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 4, fft_type::c2c, float, fft_direction::forward, false, 4, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 8, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 16, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 32, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 64, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 128, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 256, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 512, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 1024, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 2048, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 4096, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 8192, fft_type::c2c, float, fft_direction::forward, false, 16, 1>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 16384, fft_type::c2c, float, fft_direction::forward, false, 32, 1>(error_file, real_low, real_high, imag_low, imag_high);
#else   
        block_fft_precision<Arch, 2, fft_type::c2c, __half, fft_direction::forward, false, 2, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 4, fft_type::c2c, __half, fft_direction::forward, false, 4, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 8, fft_type::c2c, __half, fft_direction::forward, false, 8, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 16, fft_type::c2c, __half, fft_direction::forward, false, 8, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 32, fft_type::c2c, __half, fft_direction::forward, false, 8, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 64, fft_type::c2c, __half, fft_direction::forward, false, 8, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 128, fft_type::c2c, __half, fft_direction::forward, false, 8, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 256, fft_type::c2c, __half, fft_direction::forward, false, 8, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 512, fft_type::c2c, __half, fft_direction::forward, false, 32, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 1024, fft_type::c2c, __half, fft_direction::forward, false, 16, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 2048, fft_type::c2c, __half, fft_direction::forward, false, 16, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 4096, fft_type::c2c, __half, fft_direction::forward, false,16, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 8192, fft_type::c2c, __half, fft_direction::forward, false, 32, 2>(error_file, real_low, real_high, imag_low, imag_high);
        block_fft_precision<Arch, 16384, fft_type::c2c, __half, fft_direction::forward, false, 32, 2>(error_file, real_low, real_high, imag_low, imag_high);
#endif
        error_file.close();
    }
};

int main(int argc, char** argv) {
    double imag_low = -10.0, imag_high = 10.0;
    double real_low = -10.0, real_high = 10.0;
    std::string output_filepath;

    // Get Arguments
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "o:1:2:3:4:"))) {
        switch (opt_c) {
        case 'o':
            output_filepath = strdup(optarg);
            break;
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
        case '?':
            std::cerr << "Unknown option: " << optopt << std::endl;
            break;
        default:
            break;
        }
    }
    // Instantiate the functor with the parsed parameters
    block_fft_precision_functor<800> functor(real_low, real_high, imag_low, imag_high, output_filepath);
    // return example::sm_runner(functor);
    functor();
}
