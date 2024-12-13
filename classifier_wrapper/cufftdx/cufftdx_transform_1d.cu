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

#include "cufftdx_transform_1d.hpp"

template<unsigned int      Arch,
         unsigned int      FFTSize,
         cufftdx::fft_type FFTType,
         class PrecisionType,
         cufftdx::fft_direction FFTDirection      = cufftdx::fft_direction::forward,
         bool                   UseSuggested      = true,
         unsigned int           ElementsPerThread = 2,
         unsigned int           FFTsPerBlock      = 1>
void cufftdx_transform_1d(std::string input_file, std::string output_file) {
    using namespace cufftdx;

    using FFT_base = decltype(Block() + Type<FFTType>() + Precision<PrecisionType>() + SM<Arch>());

    using FFT_with_direction = typename std::
        conditional<FFTType == fft_type::c2c, decltype(FFT_base() + Direction<FFTDirection>()), FFT_base>::type;

    block_fft_1d<FFT_with_direction, FFTSize, ElementsPerThread, FFTsPerBlock, UseSuggested>(input_file, output_file);

    std::cout << "done\n";
    // if (verbose)
    //     std::cout << std::endl;
}

template<unsigned int Arch>
struct cufftdx_transform_1d_functor {
    unsigned int size;
    std::string input_filepath, output_filepath;
    cufftdx_transform_1d_functor(unsigned int n, std::string in, std::string out)
        :size(n), input_filepath(in), output_filepath(out) {}
    
    void operator()() { 
        using namespace cufftdx;
    // Use switch case because sizes need to be defined at compile time
#ifdef SINGLE_PRECISION
        switch (size) {
            case 2:
                cufftdx_transform_1d<Arch, 2, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 4:
                cufftdx_transform_1d<Arch, 4, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 8:
                cufftdx_transform_1d<Arch, 8, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 16:
                cufftdx_transform_1d<Arch, 16, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 32:
                cufftdx_transform_1d<Arch, 32, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 64:
                cufftdx_transform_1d<Arch, 64, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 128:
                cufftdx_transform_1d<Arch, 128, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 256:
                cufftdx_transform_1d<Arch, 256, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 512:
                cufftdx_transform_1d<Arch, 512, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 1024:
                cufftdx_transform_1d<Arch, 1024, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 2048:
                cufftdx_transform_1d<Arch, 2048, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 4096:
                cufftdx_transform_1d<Arch, 4096, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 8192:
                cufftdx_transform_1d<Arch, 8192, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            case 16384:
                cufftdx_transform_1d<Arch, 16384, fft_type::c2c, float>(input_filepath, output_filepath);
                break;
            default:
                throw std::runtime_error("Unsupported FFT size for SINGLE_PRECISION: " + std::to_string(size));
        }
#elif defined(DOUBLE_PRECISION)
        switch (size) {
            case 2:
                cufftdx_transform_1d<Arch, 2, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 4:
                cufftdx_transform_1d<Arch, 4, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 8:
                cufftdx_transform_1d<Arch, 8, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 16:
                cufftdx_transform_1d<Arch, 16, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 32:
                cufftdx_transform_1d<Arch, 32, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 64:
                cufftdx_transform_1d<Arch, 64, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 128:
                cufftdx_transform_1d<Arch, 128, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 256:
                cufftdx_transform_1d<Arch, 256, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 512:
                cufftdx_transform_1d<Arch, 512, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 1024:
                cufftdx_transform_1d<Arch, 1024, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 2048:
                cufftdx_transform_1d<Arch, 2048, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 4096:
                cufftdx_transform_1d<Arch, 4096, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            case 8192:
                cufftdx_transform_1d<Arch, 8192, fft_type::c2c, double>(input_filepath, output_filepath);
                break;
            default:
                throw std::runtime_error("Unsupported FFT size for DOUBLE_PRECISION: " + std::to_string(size));
        }
#else
        switch (size) {
            case 2:
                cufftdx_transform_1d<Arch, 2, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 4:
                cufftdx_transform_1d<Arch, 4, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 8:
                cufftdx_transform_1d<Arch, 8, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 16:
                cufftdx_transform_1d<Arch, 16, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 32:
                cufftdx_transform_1d<Arch, 32, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 64:
                cufftdx_transform_1d<Arch, 64, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 128:
                cufftdx_transform_1d<Arch, 128, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 256:
                cufftdx_transform_1d<Arch, 256, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 512:
                cufftdx_transform_1d<Arch, 512, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 1024:
                cufftdx_transform_1d<Arch, 1024, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 2048:
                cufftdx_transform_1d<Arch, 2048, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 4096:
                cufftdx_transform_1d<Arch, 4096, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 8192:
                cufftdx_transform_1d<Arch, 8192, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            case 16384:
                cufftdx_transform_1d<Arch, 16384, fft_type::c2c, __half>(input_filepath, output_filepath);
                break;
            default:
                throw std::runtime_error("Unsupported FFT size for HALF_PRECISION: " + std::to_string(size));
        }
#endif
    }
};

int main(int argc, char** argv) {
    unsigned int n;
    std::string input_filepath, output_filepath;
    // Get Arguments
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "i:o:s:")))
    {
        switch (opt_c)
        {
        case 'i':
            input_filepath = strdup(optarg);
            break;
        case 'o':
            output_filepath = strdup(optarg);
            break;
        case 's':
            n = atoi(optarg);
            break;
        case '?':
            printf("unkown option %c\n", optopt);
            break;
        default:
            break;
        }
    }
    // Instantiate the functor with the parsed parameters
    // Replace 800 with the correct Arch of your GPU
    cufftdx_transform_1d_functor<800> functor(n, input_filepath, output_filepath);
    functor();
}

