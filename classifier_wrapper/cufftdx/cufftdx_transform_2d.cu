#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <unistd.h>

#include "cufftdx_transform_2d.hpp"

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
         unsigned int           fpb_x = 1,
         unsigned int           ept_y = 2,
         unsigned int           fpb_y = 1>
void cufftdx_transform_2d(std::string input_file, std::string output_file) {
    using namespace cufftdx;

    using FFT_base = decltype(Block() + Type<FFTType>() + Precision<PrecisionType>() + SM<Arch>());

    using FFT_with_direction = typename std::
        conditional<FFTType == fft_type::c2c, decltype(FFT_base() + Direction<FFTDirection>()), FFT_base>::type;

    block_fft_2d<FFT_with_direction, FFTSizeX, FFTSizeY, ept_x, fpb_x, ept_y, fpb_y, UseSuggested>(input_file, output_file);

    std::cout << "done\n";
    // if (verbose)
    //     std::cout << std::endl;
}

template<unsigned int Arch>
struct cufftdx_transform_2d_functor {
    unsigned int nx, ny;
    std::string input_filepath, output_filepath;
    cufftdx_transform_2d_functor(unsigned int nx, unsigned int ny, std::string in, std::string out)
        :nx(nx), ny(ny), input_filepath(in), output_filepath(out) {}

    void operator()() {
        using namespace cufftdx;

        // Use switch case because sizes need to be defined at compile time
#ifdef SINGLE_PRECISION
        switch (nx) {
            case 128:
                switch (ny) {
                    case 128:
                        cufftdx_transform_2d<Arch, 128, 128, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 256:
                        cufftdx_transform_2d<Arch, 128, 256, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    default:
                        throw std::runtime_error("Unsupported ny for SINGLE_PRECISION with nx = 128: " + std::to_string(ny));
                }
                break;
            case 256:
                switch (ny) {
                    case 128:
                        cufftdx_transform_2d<Arch, 256, 128, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 256:
                        cufftdx_transform_2d<Arch, 256, 256, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 512:
                        cufftdx_transform_2d<Arch, 256, 512, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    default:
                        throw std::runtime_error("Unsupported ny for SINGLE_PRECISION with nx = 256: " + std::to_string(ny));
                }
                break;
            case 512:
                switch (ny) {
                    case 256:
                        cufftdx_transform_2d<Arch, 512, 256, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 512:
                        cufftdx_transform_2d<Arch, 512, 512, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 1024:
                        cufftdx_transform_2d<Arch, 512, 1024, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    default:
                        throw std::runtime_error("Unsupported ny for SINGLE_PRECISION with nx = 512: " + std::to_string(ny));
                }
                break;
            case 1024:
                switch (ny) {
                    case 512:
                        cufftdx_transform_2d<Arch, 1024, 512, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 1024:
                        cufftdx_transform_2d<Arch, 1024, 1024, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 2048:
                        cufftdx_transform_2d<Arch, 1024, 2048, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 8, 4>(input_filepath, output_filepath);
                        break;
                    default:
                        throw std::runtime_error("Unsupported ny for SINGLE_PRECISION with nx = 1024: " + std::to_string(ny));
                }
                break;
            case 2048:
                switch (ny) {
                    case 1024:
                        cufftdx_transform_2d<Arch, 2048, 1024, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 8, 4>(input_filepath, output_filepath);
                        break;
                    case 2048:
                        cufftdx_transform_2d<Arch, 2048, 2048, fft_type::c2c, float, fft_direction::forward, false, 16, 1, 8, 4>(input_filepath, output_filepath);
                        break;
                    default:
                        throw std::runtime_error("Unsupported ny for SINGLE_PRECISION with nx = 2048: " + std::to_string(ny));
                }
                break;
            default:
                throw std::runtime_error("Unsupported nx for SINGLE_PRECISION: " + std::to_string(nx));
        }
#elif defined(DOUBLE_PRECISION)
        switch (nx) {
            case 128:
                switch (ny) {
                    case 128:
                        cufftdx_transform_2d<Arch, 128, 128, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 256:
                        cufftdx_transform_2d<Arch, 128, 256, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    default:
                        throw std::runtime_error("Unsupported ny for DOUBLE_PRECISION with nx = 128: " + std::to_string(ny));
                }
                break;
            case 256:
                switch (ny) {
                    case 128:
                        cufftdx_transform_2d<Arch, 256, 128, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 256:
                        cufftdx_transform_2d<Arch, 256, 256, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 512:
                        cufftdx_transform_2d<Arch, 256, 512, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    default:
                        throw std::runtime_error("Unsupported ny for DOUBLE_PRECISION with nx = 256: " + std::to_string(ny));
                }
                break;
            case 512:
                switch (ny) {
                    case 256:
                        cufftdx_transform_2d<Arch, 512, 256, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 512:
                        cufftdx_transform_2d<Arch, 512, 512, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 1024:
                        cufftdx_transform_2d<Arch, 512, 1024, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    default:
                        throw std::runtime_error("Unsupported ny for DOUBLE_PRECISION with nx = 512: " + std::to_string(ny));
                }
                break;
            case 1024:
                switch (ny) {
                    case 512:
                        cufftdx_transform_2d<Arch, 1024, 512, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 1024:
                        cufftdx_transform_2d<Arch, 1024, 1024, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 16, 8>(input_filepath, output_filepath);
                        break;
                    case 2048:
                        cufftdx_transform_2d<Arch, 1024, 2048, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 8, 4>(input_filepath, output_filepath);
                        break;
                    default:
                        throw std::runtime_error("Unsupported ny for DOUBLE_PRECISION with nx = 1024: " + std::to_string(ny));
                }
                break;
            case 2048:
                switch (ny) {
                    case 1024:
                        cufftdx_transform_2d<Arch, 2048, 1024, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 8, 4>(input_filepath, output_filepath);
                        break;
                    case 2048:
                        cufftdx_transform_2d<Arch, 2048, 2048, fft_type::c2c, double, fft_direction::forward, false, 16, 1, 8, 4>(input_filepath, output_filepath);
                        break;
                    default:
                        throw std::runtime_error("Unsupported ny for DOUBLE_PRECISION with nx = 2048: " + std::to_string(ny));
                }
                break;
            default:
                throw std::runtime_error("Unsupported nx for DOUBLE_PRECISION: " + std::to_string(nx));
        }
#endif
    }
};

int main(int argc, char** argv) {
    unsigned int nx, ny;
    std::string input_filepath, output_filepath;
    // Get Arguments
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "i:o:x:y:")))
    {
        switch (opt_c)
        {
        case 'i':
            input_filepath = strdup(optarg);
            break;
        case 'o':
            output_filepath = strdup(optarg);
            break;
        case 'x':
            nx = atoi(optarg);
            break;
        case 'y':
            ny = atoi(optarg);
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
    cufftdx_transform_2d_functor<800> functor(nx, ny, input_filepath, output_filepath);
    functor();
}
