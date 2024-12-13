#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <cmath>
#include <unistd.h>
#include <omp.h>
#include <cassert>
#include <algorithm>
#include <string>
#include <cstring>
#include <iomanip>

extern char *optarg;
extern int optopt;

#ifdef SINGLE_PRECISION
typedef fftwf_complex ComplexType;
typedef float PrecType;
#elif defined(DOUBLE_PRECISION)
typedef fftw_complex ComplexType;
typedef double PrecType;
#else
#error "You must define one of SINGLE_PRECISION, DOUBLE_PRECISION, or LONGDOUBLE_PRECISION"
#endif

// Create Struct to get complex data
struct Complex {
    double real;
    double imag;
};

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

// Function to calculate the root mean square error
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
    int start_power = 1; // Starting power of 2 (2^1 = 2)
    int end_power = 27;  // Ending power of 2 (2^14 = 16384)
    
    double imag_low, imag_high, real_low, real_high;
    std::string precision, threads, output_filepath;
    imag_low = -10.0, real_low = -10.0;
    imag_high = 10.0, real_high = 10.0;
#ifdef SINGLE_PRECISION
    precision = "single";
#elif defined(DOUBLE_PRECISION)
    precision = "double";
#else 
    precision = "longdouble";
#endif

#ifdef MULTI_THREADS
    threads = "multithreads";
#else 
    threads = "singlethread";
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

    for (int i = start_power; i <= end_power; ++i) {
        N_values.push_back(1 << i); // 1 << i is equivalent to 2^i
    }

    // Create results_file and write header
    // Construct filename depending on input
    std::stringstream filename;
    filename << output_filepath
             << precision << "_prec_benchmark_imag" << double_to_string(imag_low,0) << "to" << double_to_string(imag_high,0) << "_real" << double_to_string(real_low,0) << "to" << double_to_string(real_high,0) << "_" << threads << ".csv";
    std::string filename_str = filename.str();

#ifdef SINGLE_PRECISION
    #ifdef MULTI_THREADS
    assert(fftwf_init_threads() != 0);
    std::ofstream results_file(filename_str);
    #else
    std::ofstream results_file(filename_str);
    #endif
#else 
    #ifdef MULTI_THREADS
    assert(fftw_init_threads() != 0);
    std::ofstream results_file(filename_str);
    #else
    std::ofstream results_file(filename_str);
    #endif
#endif
    if (!results_file.is_open()) {
        std::cerr << "Failed to open the results file: " << filename_str << std::endl;
        return 1;  // Return an error code if the file can't be opened
    }

    results_file << "re_low,re_high,im_low,im_high,Nx,Ny,Absolute_Error,Relative_Error[%]\n";
    for (int N : N_values) {
        // Allocate input and output arrays
        ComplexType* in_low = (ComplexType*) fftw_malloc(sizeof(ComplexType) * N);
        ComplexType* out_low = (ComplexType*) fftw_malloc(sizeof(ComplexType) * N);

        // Create a plan for the forward FFT
#ifdef DOUBLE_PRECISION
    #ifdef MULTI_THREADS
        fftw_plan_with_nthreads(omp_get_max_threads());
    #endif
        fftw_plan p = fftw_plan_dft_1d(N, in_low, out_low, FFTW_FORWARD, FFTW_ESTIMATE);   
#else 
    #ifdef MULTI_THREADS
        fftwf_plan_with_nthreads(omp_get_max_threads());
    #endif
        fftwf_plan p = fftwf_plan_dft_1d(N, in_low, out_low, FFTW_FORWARD, FFTW_ESTIMATE);
#endif
        // Get input signal
        fftwl_complex *in_long = (fftwl_complex*) fftwl_malloc(sizeof(fftwl_complex) * N);
        for(int i = 0; i < N; i++){
            in_long[i][0] = static_cast<PrecType>((static_cast<double>(rand()) / RAND_MAX) * (real_high - real_low) + real_low);
            in_long[i][1] = static_cast<PrecType>((static_cast<double>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low);
        }

        // Initialize input array with a sample signal
        for (int i = 0; i < N; ++i) {
            in_low[i][0] = static_cast<PrecType>(in_long[i][0]);
            in_low[i][1] = static_cast<PrecType>(in_long[i][1]);
            // std::cout << in[i][0] << " " << in[i][1] << std::endl;
        }

        // Execute the FFT
#ifdef DOUBLE_PRECISION
        fftw_execute(p); 
#else 
        fftwf_execute(p);
#endif

        // Get the data of the fftwl
        std::vector<Complex> data_highprecision = get_fftw_values(N, in_long);

        // Convert fft_data to double precision for comparison
        std::vector<Complex> data_lowprecision(N);
        for (size_t i = 0; i < N; ++i) {
            data_lowprecision[i].real = static_cast<double>(out_low[i][0]);
            data_lowprecision[i].imag = static_cast<double>(out_low[i][1]);
        }
        double abs_error, rel_error = 0.0;
        calculate_error(data_lowprecision, data_highprecision, &abs_error, &rel_error);

        results_file << real_low << "," << real_high << "," << imag_low << "," << imag_high << "," << N << "," << abs_error << "," << rel_error*100 << "\n";
        std::cout << "N = " << N << ", Average Error = " << abs_error << std::endl;

        // Cleanup
#ifdef DOUBLE_PRECISION
        fftw_destroy_plan(p);
        fftw_free(in_low);
        fftw_free(out_low); 
#else 
        fftwf_destroy_plan(p);
        fftwf_free(in_low);
        fftwf_free(out_low);
#endif
        

    }
    results_file.close();

    return 0;
}

