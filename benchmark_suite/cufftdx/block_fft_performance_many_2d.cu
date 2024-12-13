#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <unistd.h>

#include "block_fft_performance_2d.hpp"

extern char *optarg;
extern int optopt;

std::string double_to_string(double value, int precision = 1){
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    std::string result = ss.str();
    std::replace(result.begin(), result.end(), '.', '_');
    return result;
}

template<unsigned int Arch,
         unsigned int FFTSizeX,
         unsigned int FFTSizeY,
         cufftdx::fft_type FFTType,
         class PrecisionType,
         cufftdx::fft_direction FFTDirection      = cufftdx::fft_direction::forward,
         bool                   UseSuggested      = true,
         unsigned int           ept_x = 2,
         unsigned int           fpb_x = 2,
         unsigned int           ept_y = 1,
         unsigned int           fpb_y = 1>
void block_fft_performance_2d(std::ofstream& file, double real_low, double real_high, double imag_low, double imag_high) {
    using namespace cufftdx;

    using FFT_base = decltype(Block() + Type<FFTType>() + Precision<PrecisionType>() + SM<Arch>());

    using FFT_with_direction = typename std::
        conditional<FFTType == fft_type::c2c, decltype(FFT_base() + Direction<FFTDirection>()), FFT_base>::type;

    performance_block_fft_2d<FFT_with_direction, FFTSizeX, FFTSizeY, ept_x, fpb_x, ept_y, fpb_y, UseSuggested>(file, real_low, real_high, imag_low, imag_high);

    std::cout << "done\n";
    // if (verbose)
    //     std::cout << std::endl;
}

template<unsigned int Arch>
struct block_fft_performance_2d_functor {
    double real_low, real_high, imag_low, imag_high;
    std::string output_file;
    block_fft_performance_2d_functor(double rl, double rh, double il, double ih, std::string file)
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
                << precision << "_prec_benchmark_imag" << double_to_string(imag_low,4) << "to" << double_to_string(imag_high,4) << "_real" << double_to_string(real_low,4) << "to" << double_to_string(real_high,4) <<  ".csv";
        std::string filename_str = filename.str();

        // File to save the results
        std::ofstream file(filename_str);
        // Write header
        file << "re_low,re_high,im_low,im_high,Num_Runs,Nx,Ny,EPT_x,FPB_x,EPT_y,FPB_y,GFLOPS,AverageTime(s),Power(W),AverageEnergy(mJ)\n";

        // To specify EPT and FPB values, set UsedSuggested to false.
        // FFTDirection is used if and only if FFTType is C2C.
#ifdef DOUBLE_PRECISION
        block_fft_performance_2d<Arch, 128, 128, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 256, 128, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 128, 256, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 256, 256, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 512, 256, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 256, 512, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 512, 512, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 1024, 512, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 512, 1024, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 1024, 1024, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 2048, 1024, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 8, 4>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 1024, 2048, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 8, 4>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 2048, 2048, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 8, 4>(file, real_low, real_high, imag_low, imag_high);
        // block_fft_performance_2d<Arch, 256, 64, fft_type::c2c, double, fft_direction::forward, false, 4, 4, 1, 1>(file, real_low, real_high, imag_low, imag_high);
        // block_fft_performance_2d<Arch, 256, 128, fft_type::c2c, double, fft_direction::forward, false, 4, 4, 1, 1>(file, real_low, real_high, imag_low, imag_high);
        // block_fft_performance_2d<Arch, 256, 256, fft_type::c2c, double, fft_direction::forward, false, 4, 4, 1, 1>(file, real_low, real_high, imag_low, imag_high);
        // block_fft_performance_2d<Arch, 256, 512, fft_type::c2c, double, fft_direction::forward, false, 4, 4, 1, 1>(file, real_low, real_high, imag_low, imag_high);
        // block_fft_performance_2d<Arch, 256, 1024, fft_type::c2c, double, fft_direction::forward, false, 4, 4, 1, 1>(file, real_low, real_high, imag_low, imag_high);

#elif defined(SINGLE_PRECISION)
        block_fft_performance_2d<Arch, 128, 128, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 256, 128, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 128, 256, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 256, 256, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 512, 256, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 256, 512, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 512, 512, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 1024, 512, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 512, 1024, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 1024, 1024, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 2048, 1024, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 8, 4>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 1024, 2048, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 8, 4>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 2048, 2048, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 8, 4>(file, real_low, real_high, imag_low, imag_high);
        block_fft_performance_2d<Arch, 4096, 4096, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 8, 2>(file, real_low, real_high, imag_low, imag_high);
#else  
       #error "You must define either SINGLE_PRECISION OR DOUBLE_PRECISION"
#endif
        file.close();
    }
};

int main(int argc, char** argv) {
    double imag_low = -10.0, imag_high = 10.0;
    double real_low = -10.0, real_high = 10.0;
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
    block_fft_performance_2d_functor<800> functor(real_low, real_high, imag_low, imag_high, output_filepath);
    // return example::sm_runner(functor);
    functor();
}
