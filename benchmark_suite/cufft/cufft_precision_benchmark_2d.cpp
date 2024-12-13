#include <complex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cufft.h>
#include <cstdio>
#include <algorithm>
#include <string>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cufftXt.h>
#include "fftw3.h"
#include <unistd.h>

extern char *optarg;
extern int optopt;

typedef half2 ftype;

#ifdef SINGLE_PRECISION
typedef cufftComplex ComplexType;
typedef float PrecType;
#elif defined(DOUBLE_PRECISION)
typedef cufftDoubleComplex ComplexType;
typedef double PrecType;
#elif defined(HALF_PRECISION)
typedef ftype ComplexType; 
#else
#error "You must define one of SINGLE_PRECISION, DOUBLE_PRECISION, or HALF_PRECISION"
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
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(status);                                                                          \
        }                                                                                               \
    }
#endif // CUDA_CHECK_AND_EXIT

// Create Struct to get complex data
struct Complex {
    double real;
    double imag;
};

// Calculate fftw
std::vector<Complex> get_fftw_values(int Nx, int Ny, fftwl_complex* in){
    fftwl_complex* out = (fftwl_complex*) fftwl_malloc(sizeof(fftwl_complex) * Nx * Ny);
    fftwl_plan p = fftwl_plan_dft_2d(Nx, Ny, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwl_execute(p);
    // Read in output vector
    std::vector<Complex> data_highprecision(Nx*Ny);
    for (size_t i = 0; i < (Nx*Ny); ++i) {
        data_highprecision[i].real = static_cast<double>(out[i][0]);
        data_highprecision[i].imag = static_cast<double>(out[i][1]);
    }
    fftwl_destroy_plan(p);
    fftwl_free(in);
    fftwl_free(out);
    return data_highprecision;
}

// Function to calculate the error
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

std::vector<std::pair<int, int>> generateNvalues(int start, int max) {
    std::vector<int> powersOfTwo;
    std::vector<std::pair<int, int>> sizes;
    // Generate all powers of two from 2^start to 2^max
    for (int i = start; i <= max; ++i) {
        powersOfTwo.push_back(1 << i); // Equivalent to 2^i
    }
    // Generate square and rectangular sizes
    for (int size : powersOfTwo) {
        // Add square size
        sizes.push_back({size, size});
        // Add rectangular sizes with two powers of two difference
        for (int otherSize : powersOfTwo) {
            int diff = std::abs(std::log2(size) - std::log2(otherSize));
            if (diff == 2 || diff == 1) {
                sizes.push_back({size, otherSize});
                sizes.push_back({otherSize, size});
            }
        }
    }
    // Remove duplicates by sorting and using unique
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    return sizes;
}

std::vector<int> generateSizes(int minSize, int maxSize, double factor) {
    std::vector<int> sizes;
    int currentSize = minSize;

    while (currentSize <= maxSize) {
        sizes.push_back(currentSize);
        currentSize = static_cast<int>(std::round(currentSize * factor));
    }

    return sizes;
}

std::string double_to_string(double value, int precision = 1){
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    std::string result = ss.str();
    std::replace(result.begin(), result.end(), '.', '_');
    return result;
}

int main(int argc, char *argv[]) {
    // Different values of N (number of sample points) to test
    std::vector<std::pair<int,int>> N_values;
    int start_power = 7;
    int end_power = 11;
    
    double imag_low, imag_high, real_low, real_high;
    std::string precision, output_filepath;
    imag_low = -10.0, real_low = -10.0;
    imag_high = 10.0, real_high = 10.0;

#ifdef SINGLE_PRECISION
    precision = "single";
#elif defined(DOUBLE_PRECISION)
    precision = "double";
#else 
    precision = "half";
#endif
    // Get Arguments
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "e:o:1:2:3:4:")))
    {
        switch (opt_c)
        {
        case 'e':
            end_power = atoi(optarg);
            break;
        case 'o':
            output_filepath = strdup(optarg);
            break;
        case '3':
            imag_low = atof(optarg);
            break;
        case '4':
            imag_high = atof(optarg);
            break;
        case '1':
            real_low = atof(optarg);
            break;
        case '2':
            real_high = atof(optarg);
            break;   
        case '?':
            printf("unkown option %c\n", optopt);
            break;
        default:
            break;
        }
    }

    N_values = generateNvalues(start_power, end_power);
    // Construct filename depending on input
    std::stringstream filename;
    filename << output_filepath
             << precision << "_prec_benchmark_imag" << double_to_string(imag_low,4) << "to" << double_to_string(imag_high,4) << "_real" << double_to_string(real_low,4) << "to" << double_to_string(real_high,4) << ".csv";
    std::string filename_str = filename.str();

    // CSV file to save the results
#ifdef SINGLE_PRECISION
    std::ofstream results_file(filename_str);
#elif defined(DOUBLE_PRECISION)
    std::ofstream results_file(filename_str);
#else
    std::ofstream results_file(filename_str);
#endif
    // results_file << "re_low,re_high,im_low,im_high,Nx,Ny,Absolute_Error,Relative_Error[%]\n";
    results_file << "variance_real,variance_imag,difference_real_imag,Magnitude,Nx,Ny,Absolute_Error,Relative_Error[%]\n";


    size_t ws = 0;

    for (const auto& size : N_values) {
        int Nx = size.first;
        int Ny = size.second;
        long long n[2] = {Nx, Ny};        

        // Allocate host memory for the input array
        ComplexType *h_input = (ComplexType *)malloc(sizeof(ComplexType) * Nx * Ny);
        ComplexType *h_output = (ComplexType *)malloc(sizeof(ComplexType) * Nx * Ny);
        
        // Get input signal
        fftwl_complex *in = (fftwl_complex*) fftwl_malloc(sizeof(fftwl_complex) * Nx * Ny);
        double real_mean = 0.0, imag_mean = 0.0, difference_mean = 0.0, magnitude_mean = 0.0;
        for(int i = 0; i < (Nx*Ny); i++){
            in[i][0] = (static_cast<long double>(rand()) / RAND_MAX) * (real_high - real_low) + real_low;
            in[i][1] = (static_cast<long double>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low;
            // Get mean values
            real_mean += in[i][0];
            imag_mean += in[i][1];
            difference_mean += std::abs(in[i][0] - in[i][1]);
            magnitude_mean += std::sqrt(in[i][0]*in[i][0] + in[i][1]*in[i][1]);
        }
        difference_mean /= (Nx*Ny);
        magnitude_mean /= (Nx*Ny);
        real_mean /= (Nx*Ny);
        imag_mean /= (Nx*Ny);

        // Calculate Variance for imaginary and real part
        double variance_real = 0.0, variance_imag = 0.0;
        for(int i = 0; i < (Nx*Ny); i++){
            variance_real += std::pow(in[i][0] - real_mean, 2);
            variance_imag += std::pow(in[i][1] - imag_mean, 2);
        }    
        variance_real /= (Nx*Ny);
        variance_imag /= (Nx*Ny); 

        // Initialize input array with a sample signal
        for (int i = 0; i < (Nx*Ny); ++i) {
#ifdef HALF_PRECISION
            h_input[i].x = __double2half(in[i][0]);
            h_input[i].y = __double2half(in[i][1]);
#else
            h_input[i].x = static_cast<PrecType>(in[i][0]);
            h_input[i].y = static_cast<PrecType>(in[i][1]);
#endif
        }
        // Allocate device memory for the input and output arrays
        ComplexType *d_input;
        ComplexType *d_output;
        cudaMalloc((void **)&d_input, sizeof(ComplexType) * Nx * Ny);
        cudaMalloc((void **)&d_output, sizeof(ComplexType) * Nx * Ny);

        // Copy the input data from host to device
        cudaMemcpy(d_input, h_input, sizeof(ComplexType) * Nx * Ny, cudaMemcpyHostToDevice);        

        // Create a plan for the forward FFT
        cufftHandle plan;
        CUFFT_CHECK_AND_EXIT(cufftCreate(&plan));
#ifdef SINGLE_PRECISION
        CUFFT_CHECK_AND_EXIT(cufftPlan2d(&plan, Nx, Ny, CUFFT_C2C));
#elif defined(DOUBLE_PRECISION)
        CUFFT_CHECK_AND_EXIT(cufftPlan2d(&plan, Nx, Ny, CUFFT_Z2Z));
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
        CUDA_CHECK_AND_EXIT(cudaMemcpy(h_output, d_output, sizeof(ComplexType) * Nx * Ny, cudaMemcpyDeviceToHost));

        // Get the data of the file
        std::vector<Complex> data_highprecision = get_fftw_values(Nx, Ny, in);

        // Convert fft_data to double precision for comparison
        std::vector<Complex> data_lowprecision(Nx*Ny);
        for (size_t i = 0; i < (Nx*Ny); ++i) {
            data_lowprecision[i].real = static_cast<double>(h_output[i].x);
            data_lowprecision[i].imag = static_cast<double>(h_output[i].y);
        }
        double rel_error,abs_error = 0.0;
        calculate_error(data_lowprecision, data_highprecision, &abs_error, &rel_error);

        results_file << variance_real << "," << variance_imag << "," << difference_mean << "," << magnitude_mean << "," << Nx << "," << Ny << "," << abs_error << "," << rel_error*100 << "\n";
        std::cout << "Nx = " << Nx << ", Ny = " << Ny << ", Average Error = " << rel_error*100 << "%" << "\n";

        // Clean up
        CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));
        CUDA_CHECK_AND_EXIT(cudaFree(d_input));
        CUDA_CHECK_AND_EXIT(cudaFree(d_output));
        free(h_input);
        free(h_output);

    }
    results_file.close();

    return 0;
}

