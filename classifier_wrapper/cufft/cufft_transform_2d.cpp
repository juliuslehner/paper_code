#include <complex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cufft.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <iomanip>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cufftXt.h>
#include "fftw3.h"
#include <unistd.h>
using namespace std;

extern char *optarg;
extern int optopt;

typedef half2 ftype;
typedef nv_bfloat162 bfloat;

#ifdef SINGLE_PRECISION
typedef cufftComplex ComplexType;
typedef float PrecType;
#elif defined(DOUBLE_PRECISION)
typedef cufftDoubleComplex ComplexType;
typedef double PrecType;
#elif defined(HALF_PRECISION)
typedef ftype ComplexType; 
#elif defined(BFLOAT_PRECISION)
typedef bfloat ComplexType;
#else
#error "You must define one of SINGLE_PRECISION, DOUBLE_PRECISION, BFLOAT_PRECISION or HALF_PRECISION"
#endif

#ifndef CUFFT_CHECK_AND_EXIT
#define CUFFT_CHECK_AND_EXIT( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CHECK_AND_EXIT

// cuda error checking
#ifndef CUDA_CHECK_AND_EXIT
#define CUDA_CHECK_AND_EXIT(error)                                                                      \
    {                                                                                                   \
        auto status = static_cast<cudaError_t>(error);                                                  \
        if (status != cudaSuccess) {                                                                    \
            cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << endl; \
            exit(status);                                                                          \
        }                                                                                               \
    }
#endif // CUDA_CHECK_AND_EXIT

// Function to read binary data from file
// File is expected to be interleaved complex data in double
void read_binary(const string& filename, ComplexType* input, int n) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }

    // Read interleaved binary data
    double real, imag;
    for (int i = 0; i < n; ++i) {
        file.read(reinterpret_cast<char*>(&real), sizeof(double));
        file.read(reinterpret_cast<char*>(&imag), sizeof(double));
        if (!file) {
            throw std::runtime_error("Unexpected end of file while reading: " + filename);
        }

#ifdef HALF_PRECISION
        input[i].x = __double2half(real);
        input[i].y = __double2half(imag);
#else
        input[i].x = static_cast<PrecType>(real);
        input[i].y = static_cast<PrecType>(imag);
#endif
    }
    file.close();
}

void write_binary(const std::string& filepath, ComplexType* output, int n) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }

    // Write the real and imaginary parts interleaved
    for (int i = 0; i < n; ++i) {
        double real = static_cast<double>(output[i].x);
        double imag = static_cast<double>(output[i].y);
        file.write(reinterpret_cast<const char*>(&real), sizeof(double));
        file.write(reinterpret_cast<const char*>(&imag), sizeof(double)); 
    }

    // Close the file
    file.close();
}

int main(int argc, char *argv[]) {
    int nx, ny;
    string input_filepath, output_filepath;
    string precision, threads, signal_filename;
#ifdef SINGLE_PRECISION
    precision = "single";
#elif defined(DOUBLE_PRECISION)
    precision = "double";
#else 
    precision = "half";
#endif
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

    size_t ws = 0;
    long long n[2] = {nx, ny}; 

    // Allocate host memory for the input array
    ComplexType *h_input = (ComplexType *)malloc(sizeof(ComplexType) * nx * ny);
    ComplexType *h_output = (ComplexType *)malloc(sizeof(ComplexType) * nx * ny);
    read_binary(input_filepath, h_input, nx*ny);

    // Allocate device memory for the input and output arrays
    ComplexType *d_input;
    ComplexType *d_output;
    cudaMalloc((void **)&d_input, sizeof(ComplexType) * nx * ny);
    cudaMalloc((void **)&d_output, sizeof(ComplexType) * nx * ny);

    // Copy the input data from host to device
    cudaMemcpy(d_input, h_input, sizeof(ComplexType) * nx * ny, cudaMemcpyHostToDevice);        

    // Create a plan for the forward FFT
    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftCreate(&plan));
#ifdef SINGLE_PRECISION
    CUFFT_CHECK_AND_EXIT(cufftPlan2d(&plan, nx, ny, CUFFT_C2C));
#elif defined(DOUBLE_PRECISION)
    CUFFT_CHECK_AND_EXIT(cufftPlan2d(&plan, nx, ny, CUFFT_Z2Z));
#else
    CUFFT_CHECK_AND_EXIT(cufftXtMakePlanMany(plan, 2, n, NULL, 1, 1, CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, &ws, CUDA_C_16F));
#endif

    // Execute the FFT
#ifdef SINGLE_PRECISION
    CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD));
#elif defined(DOUBLE_PRECISION)
    CUFFT_CHECK_AND_EXIT(cufftExecZ2Z(plan, d_input, d_output, CUFFT_FORWARD));
#else
    cufftXtExec(plan, d_input, d_output, CUFFT_FORWARD);
#endif
    // Copy the result from device to host
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_output, d_output, sizeof(ComplexType) * nx * ny, cudaMemcpyDeviceToHost));

    // Write results to the output file
    write_binary(output_filepath, h_output, nx*ny);

    cout << "Transform completed" << "\n";
    cout << "Results saved to " << output_filepath << "\n";

    // Clean up
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));
    CUDA_CHECK_AND_EXIT(cudaFree(d_input));
    CUDA_CHECK_AND_EXIT(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}

