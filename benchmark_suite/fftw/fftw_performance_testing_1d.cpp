#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <omp.h>
#include <cassert>
#include <papi.h>
#include <algorithm>
#include <string>
#include <cstring>
#include <iomanip>

#define NUM_EVENTS 4
extern char *optarg;
extern int optopt;

#ifdef SINGLE_PRECISION
typedef fftwf_complex ComplexType;
typedef float PrecType;
#elif defined(DOUBLE_PRECISION)
typedef fftw_complex ComplexType;
typedef double PrecType;
#elif defined(LONGDOUBLE_PRECISION)
typedef fftwl_complex ComplexType;
typedef long double PrecType;
#else
#error "You must define one of SINGLE_PRECISION, DOUBLE_PRECISION, or LONGDOUBLE_PRECISION"
#endif

struct Complex {
    double real;
    double imag;
};

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
    int start_power = 1; // Starting power of 2 (2^1 = 2)
    int end_power = 23;  // Ending power of 2 (2^14 = 16384)

    // Number of runs to average the timing per plan execution
    int num_runs_per_plan = 1000;
    int warmup_runs = 10;
    int num_repeats = 1;

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
    while (EOF != (opt_c = getopt(argc, argv, "n:s:e:r:w:o:3:4:1:2:")))
    {
        switch (opt_c)
        {
        case 'n':
            num_repeats = atoi(optarg);
            break;
        case 's':
            start_power = atoi(optarg);
            break;
        case 'e':
            end_power = atoi(optarg);
            break;
        case 'r':
            num_runs_per_plan = atoi(optarg);
            break;
        case 'w':
            warmup_runs = atoi(optarg);
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
             << precision << "_prec_benchmark_imag" << double_to_string(imag_low,0) << "to" << double_to_string(imag_high,0) << "_real" << double_to_string(real_low,0) << "to" << double_to_string(real_high,0) << "_" << threads << ".csv";
    std::string filename_str = filename.str();

    // CSV file to save the results
#ifdef SINGLE_PRECISION
    #ifdef MULTI_THREADS
        assert(fftwf_init_threads() != 0);
        std::ofstream results_file(filename_str);
    #else
        std::ofstream results_file(filename_str);
    #endif
#elif defined(DOUBLE_PRECISION)
    #ifdef MULTI_THREADS
        assert(fftw_init_threads() != 0);
        std::ofstream results_file(filename_str);
    #else
        std::ofstream results_file(filename_str);
    #endif
#else
    #ifdef MULTI_THREADS
        assert(fftwl_init_threads() != 0);
        std::ofstream results_file(filename_str);
    #else
        std::ofstream results_file(filename_str);
    #endif
#endif
    if (!results_file.is_open()) {
        std::cerr << "Failed to open the results file: " << filename_str << std::endl;
        return 1;  // Return an error code if the file can't be opened
    }
    results_file << "re_low,re_high,im_low,im_high,N,Number_Runs,Num_Repeats,GFLOPS,Total_duration(s),Average_duration(s),Power(W),Average_Energy(mJ)\n";

    //PAPI Inits
    int k;
    char RAPLEventNames[NUM_EVENTS][PAPI_MAX_STR_LEN] = {"rapl:::PACKAGE_ENERGY:PACKAGE0"};
    int RAPLEventSet = PAPI_NULL;
    long long RAPLValues[NUM_EVENTS];
    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&RAPLEventSet);
    for(k=0; k<NUM_EVENTS; k++)
    {
        PAPI_add_named_event(RAPLEventSet, RAPLEventNames[k]);
    }

    for (int N : N_values) {
        double total_gflops = 0.0;
        double total_total_duration = 0.0;
        double total_avg_duration = 0.0;
        double total_power = 0.0;
        double total_avg_energy = 0.0;
        for (int repeat = 0; repeat < num_repeats; ++repeat) {

            // Allocate input and output arrays
            ComplexType* in = (ComplexType*) fftw_malloc(sizeof(ComplexType) * N);
            ComplexType* out = (ComplexType*) fftw_malloc(sizeof(ComplexType) * N);

            // Create a plan for the forward FFT
#ifdef DOUBLE_PRECISION
    #ifdef MULTI_THREADS
            fftw_plan_with_nthreads(omp_get_max_threads());
    #endif
            fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);   
#elif defined(SINGLE_PRECISION) 
    #ifdef MULTI_THREADS
            fftwf_plan_with_nthreads(omp_get_max_threads());
    #endif
            fftwf_plan p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
#else
    #ifdef MULTI_THREADS
            fftwl_plan_with_nthreads(omp_get_max_threads());
    #endif
            fftwl_plan p = fftwl_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);   
#endif
            // Initialize input array with a sample signal
            for (int i = 0; i < N; ++i) {
                in[i][0] = static_cast<PrecType>((static_cast<double>(rand()) / RAND_MAX) * (real_high - real_low) + real_low);
                in[i][1] = static_cast<PrecType>((static_cast<double>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low);
            }

            // Measure the time taken for each FFT execution and sum the durations
            std::chrono::duration<double> total_duration = std::chrono::duration<double>::zero();
            for (int run = 0; run < warmup_runs; ++run){
#ifdef DOUBLE_PRECISION
                fftw_execute(p); 
#elif defined(SINGLE_PRECISION)
                fftwf_execute(p);
#else
                fftwl_execute(p);                    
#endif                
            }
            PAPI_start(RAPLEventSet);
            auto start = std::chrono::high_resolution_clock::now();
            for (int run = 0; run < (num_runs_per_plan - warmup_runs); ++run) {
#ifdef DOUBLE_PRECISION
                    fftw_execute(p); 
#elif defined(SINGLE_PRECISION)
                    fftwf_execute(p);
#else
                    fftwl_execute(p);                    
#endif                    
            }
            auto end = std::chrono::high_resolution_clock::now();
            PAPI_read(RAPLEventSet, RAPLValues);
            PAPI_stop(RAPLEventSet, RAPLValues);
            total_duration = end - start;
            double total_energy = static_cast<double>(RAPLValues[0]);

            // Calculate GFLOPS
            double gflops = 1.0 * (num_runs_per_plan - warmup_runs) * 5.0 * N *
                    (std::log(N) / std::log(2)) / total_duration.count() / 1000000000.0;

            // Calculate the average duration (in s)
            double avg_duration = total_duration.count() / (num_runs_per_plan - warmup_runs);
            double avg_energy = total_energy / (num_runs_per_plan - warmup_runs);
            // Calculate Power
            double power = (double)(total_energy*1.0E-9/total_duration.count());
            
            // Get repeat data
            total_gflops += gflops;
            total_total_duration += total_duration.count();
            total_avg_duration += avg_duration;
            total_power += power;
            total_avg_energy += avg_energy;

            // Output the total and average time taken for the FFT execution
            std::cout << "N = " << N << ", Repeat " << repeat + 1 << ": Total FFT execution time for " << num_runs_per_plan
                      << " runs: " << total_duration.count() << " seconds" << std::endl;
            std::cout << "Average FFT execution time per run: " << avg_duration << " seconds, " << power << "W" << std::endl;
            // printf("PAPI Counters:\nCPU: %lld[nJ]\n", RAPLValues[0]);

            // Cleanup
#ifdef DOUBLE_PRECISION
            fftw_destroy_plan(p);
            fftw_free(in);
            fftw_free(out); 
#elif defined(SINGLE_PRECISION)
            fftwf_destroy_plan(p);
            fftwf_free(in);
            fftwf_free(out);
#else       
            fftwl_destroy_plan(p);
            fftwl_free(in);
            fftwl_free(out);
#endif
        }
        // Save the results to the CSV file
        results_file << real_low << "," << real_high << "," << imag_low << "," << imag_high << "," << N << "," << num_runs_per_plan 
                     << "," << num_repeats << "," << total_gflops/num_repeats << "," << total_total_duration/num_repeats << "," << total_avg_duration/num_repeats << "," << total_power/num_repeats << "," << total_avg_energy/num_repeats << "\n";
    }

    // Close the CSV file
    results_file.close();

    return 0;
}

