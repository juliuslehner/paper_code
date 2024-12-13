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

std::vector<Complex> read_csv(const std::string& filename, int N) {
    std::vector<Complex> data;
    std::ifstream file(filename);
    std::string line;
    // Skip the header
    std::getline(file, line);
    // Read data
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        // Read N value from the line
        std::getline(iss, token, ',');
        int currentN = std::stoi(token);
        // If currentN doesn't match the desired N, skip to the next iteration
        if (currentN != N) {
            continue;
        }
        // Read Real and Imag parts
        Complex c;
        std::getline(iss, token, ',');  // Skip the Index column
        std::getline(iss, token, ',');
        c.real = std::stod(token);
        std::getline(iss, token, ',');
        c.imag = std::stod(token);
        // Add the Complex number to the data vector
        data.push_back(c);
        // Keep reading as long as the next row has the same N value
        while (std::getline(file, line)) {
            std::istringstream iss_next(line);
            std::getline(iss_next, token, ',');
            int nextN = std::stoi(token);

            if (nextN != N) {
                break;
            }
            // If it's the same N, process this line as well
            std::getline(iss_next, token, ',');  // Skip the Index column
            std::getline(iss_next, token, ',');
            c.real = std::stod(token);
            std::getline(iss_next, token, ',');
            c.imag = std::stod(token);
            // Add the Complex number to the data vector
            data.push_back(c);
        }

        break;
    }

    return data;
}

// Calculate fftw
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
    std::vector<int> N_values;
    int start_power = 10; // Starting power of 2 (2^1 = 2)
    int end_power = 20;  // Ending power of 2 (2^14 = 16384)

    double imag_low, imag_high, real_low, real_high;
    std::string precision, threads, output_filepath;
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
    while (EOF != (opt_c = getopt(argc, argv, "s:e:o:1:2:3:4:")))
    {
        switch (opt_c)
        {
        case 's':
            start_power = atoi(optarg);
            break;
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

    // int minSize = (int)(pow(2, start_power));
    // int maxSize = (int)(pow(2, end_power));
    // double factor = 1.4;
    // N_values = generateSizes(minSize, maxSize, factor);
    for (int i = start_power; i <= end_power; ++i) {
        N_values.push_back(1 << i); // 1 << i is equivalent to 2^i
    }
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
    if (!results_file.is_open()) {
        std::cerr << "Failed to open the results file: " << filename_str << std::endl;
        return 1;  // Return an error code if the file can't be opened
    }
    // results_file << "re_low,re_high,im_low,im_high,N,Absolute_Error,Relative_Error[%]\n";
    results_file << "variance_real,variance_imag,variance_diff,rate_of_change,Magnitude,N,Absolute_Error,Relative_Error[%]\n";

    size_t ws = 0;
    for (int N : N_values) {
        long long sig_size = N;

        // Allocate host memory for the input array
        ComplexType *h_input = (ComplexType *)malloc(sizeof(ComplexType) * N);
        ComplexType *h_output = (ComplexType *)malloc(sizeof(ComplexType) * N);
        
        // std::vector<Complex> input_data;
        // input_data = read_csv(signal_filename, N);
        // Get input signal
        fftwl_complex *in = (fftwl_complex*) fftwl_malloc(sizeof(fftwl_complex) * N);
        double real_mean = 0.0, imag_mean = 0.0, rate_change = 0.0 , magnitude_mean = 0.0;
        for(int i = 0; i < N; i++){
            in[i][0] = (static_cast<long double>(rand()) / RAND_MAX) * (real_high - real_low) + real_low;
            in[i][1] = (static_cast<long double>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low;
            // in[i][0] = input_data[i].real;
            // in[i][1] = input_data[i].imag;
            // Get mean values
            real_mean += in[i][0];
            imag_mean += in[i][1];
            if (i > 0) rate_change += std::abs(std::sqrt(in[i][0]*in[i][0] + in[i][1]*in[i][1]) - std::sqrt(in[i-1][0]*in[i-1][0] + in[i-1][1]*in[i-1][1]));
            magnitude_mean += std::sqrt(in[i][0]*in[i][0] + in[i][1]*in[i][1]);
        }
        rate_change /= (N-1);
        magnitude_mean /= N;
        real_mean /= N;
        imag_mean /= N;

        // Calculate Variance for imaginary and real part
        double variance_real = 0.0, variance_imag = 0.0;
        for(int i = 0; i < N; i++){
            variance_real += std::pow(in[i][0] - real_mean, 2);
            variance_imag += std::pow(in[i][1] - imag_mean, 2);
        }    
        variance_real /= N;
        variance_imag /= N;    
        double var_diff = std::abs(variance_imag - variance_real);

        // Initialize input array with a sample signal
        for (int i = 0; i < N; ++i) {
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
        cudaMalloc((void **)&d_input, sizeof(ComplexType) * N);
        cudaMalloc((void **)&d_output, sizeof(ComplexType) * N);

        // Copy the input data from host to device
        cudaMemcpy(d_input, h_input, sizeof(ComplexType) * N, cudaMemcpyHostToDevice);        

        // Create a plan for the forward FFT
        cufftHandle plan;
        CUFFT_CHECK_AND_EXIT(cufftCreate(&plan));
#ifdef SINGLE_PRECISION
        CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
#elif defined(DOUBLE_PRECISION)
        CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));
#elif defined(BFLOAT_PRECISION)
        CUFFT_CHECK_AND_EXIT(cufftXtMakePlanMany(plan, 1, &sig_size, NULL, 1, 1, CUDA_C_16BF, NULL, 1, 1, CUDA_C_16BF, 1, &ws, CUDA_C_16BF));
#else
        CUFFT_CHECK_AND_EXIT(cufftXtMakePlanMany(plan, 1, &sig_size, NULL, 1, 1, CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, &ws, CUDA_C_16F));
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
        CUDA_CHECK_AND_EXIT(cudaMemcpy(h_output, d_output, sizeof(ComplexType) * N, cudaMemcpyDeviceToHost));

        // Get the data of the file
        std::vector<Complex> data_highprecision = get_fftw_values(N, in);

        // Convert fft_data to double precision for comparison
        std::vector<Complex> data_lowprecision(N);
        for (size_t i = 0; i < N; ++i) {
            data_lowprecision[i].real = static_cast<double>(h_output[i].x);
            data_lowprecision[i].imag = static_cast<double>(h_output[i].y);
        }
        double rel_error,abs_error = 0.0;
        calculate_error(data_lowprecision, data_highprecision, &abs_error, &rel_error);

        // results_file << real_low << "," << real_high << "," << imag_low << "," << imag_high << "," << N << "," << abs_error << "," << rel_error*100 << "\n";
        results_file << variance_real << "," << variance_imag << "," << var_diff << "," << rate_change << "," << magnitude_mean << "," << N << "," << abs_error << "," << rel_error*100 << "\n";
        std::cout << "Var_Real: " << variance_real << ", Var_Imag: " << variance_imag << ", Difference: " << var_diff << ", Magnitude: " << magnitude_mean << ", N = " << N << ", Average Error = " << rel_error*100 << "%" << "\n";

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

