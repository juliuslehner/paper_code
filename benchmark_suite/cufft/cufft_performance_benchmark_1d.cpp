#include <complex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cufft.h>
#include <cstdio>
#include <cufftXt.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <iomanip>
#include <cuda_fp16.h>
#include <unistd.h>
#include <cuda_bf16.h>

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
#error "You must define one of SINGLE_PRECISION, DOUBLE_PRECISION, BFLOAT_PRECISION, or HALF_PRECISION"
#endif

#ifndef CUFFT_CHECK_AND_EXIT
#define CUFFT_CHECK_AND_EXIT(call)                                                         \
    {                                                                                      \
        auto status = static_cast<cufftResult>(call);                                      \
        if (status != CUFFT_SUCCESS)                                                       \
            fprintf(stderr,                                                                \
                    "ERROR: CUFFT call \"%s\" in line %d of file %s failed "               \
                    "with code (%d).\n",                                                   \
                    #call,                                                                 \
                    __LINE__,                                                              \
                    __FILE__,                                                              \
                    status);                                                               \
    }
#endif // CUFFT_CHECK_AND_EXIT

#ifndef CUDA_CHECK_AND_EXIT
#define CUDA_CHECK_AND_EXIT(error)                                                        \
    {                                                                                     \
        auto status = static_cast<cudaError_t>(error);                                    \
        if (status != cudaSuccess) {                                                      \
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                       \
            std::exit(status);                                                            \
        }                                                                                 \
    }
#endif // CUDA_CHECK_AND_EXIT

struct Complex {
    double real;
    double imag;
};

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
    int end_power = 25;  // Ending power of 2 (2^14 = 16384)

    // Number of runs to average the timing per plan execution
    int num_runs_per_plan = 10000;
    int warmup_runs = 10;
    int num_repeats = 1;
    int number = 1;

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
    while (EOF != (opt_c = getopt(argc, argv, "b:s:e:r:w:n:o:0:1:2:3:4:")))
    {
        switch (opt_c)
        {
        case 'b':
            number = atoi(optarg);
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
        case 'n':
            num_repeats = atoi(optarg);
            break;
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
            printf("unkown option %c\n", optopt);
            break;
        default:
            break;
        }
    }

    for (int i = start_power; i <= end_power; ++i) {
        N_values.push_back(1 << i); // 1 << i is equivalent to 2^i
    }

    // Construct filename depending on input
    std::stringstream filename;
    filename << output_filepath
             << precision << "_prec_benchmark_imag" << double_to_string(imag_low,0) << "to" << double_to_string(imag_high,0) << "_real" << double_to_string(real_low,0) << "to" << double_to_string(real_high,0) << ".csv";
    std::string filename_str = filename.str();

    // CSV file to save the results
#ifdef SINGLE_PRECISION
    std::ofstream results_file(filename_str);
#elif defined(DOUBLE_PRECISION)
    std::ofstream results_file(filename_str);
#else
    std::ofstream results_file(filename_str);
#endif

    results_file << "re_low,re_high,im_low,im_high,N,Batches,Num_Runs,Num_Repeats,GFLOPS,AverageTime(s),Power(W),Average_Energy(mJ)\n";
    // results_file << "variance_real,variance_imag,difference_real_imag,Magnitude,N,Absolute_Error,Relative_Error[%]\n";  


    for (int N : N_values) {
        double total_gflops = 0.0;
        double total_avg_duration = 0.0;
        double total_power = 0.0;
        double total_avg_energy = 0.0;
        int batch_size = static_cast<int>((1<<end_power)/N);
        long long sig_size = N;
        for (int repeat = 0; repeat < num_repeats; ++repeat) {
            // Initialize NVML
            nvmlInit();

            // Create a CUDA stream
            cudaStream_t stream;
            CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));
            // Allocate host memory for the input array
            ComplexType *h_input = (ComplexType *)malloc(sizeof(ComplexType) * N * batch_size);
    
            // Initialize input array with a sample signal
            // std::vector<Complex> input = read_csv(inputFilePath);
            // double real_mean = 0.0, imag_mean = 0.0, difference_mean = 0.0, magnitude_mean = 0.0;
            for (int i = 0; i < N; ++i) {
#ifdef HALF_PRECISION
            h_input[i].x = __double2half((static_cast<double>(rand()) / RAND_MAX) * (real_high - real_low) + real_low);
            h_input[i].y = __double2half((static_cast<double>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low);
#else
            h_input[i].x = static_cast<PrecType>((static_cast<double>(rand()) / RAND_MAX) * (real_high - real_low) + real_low);
            h_input[i].y = static_cast<PrecType>((static_cast<double>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low);
#endif  
            // Get mean values
            // real_mean += h_input[i].x;
            // imag_mean += in[i][1];
            // difference_mean += std::abs(in[i][0] - in[i][1]);
            // magnitude_mean += std::sqrt(in[i][0]*in[i][0] + in[i][1]*in[i][1]);

            }

            // Allocate device memory for the input and output arrays
            ComplexType *d_input;
            ComplexType *d_output;
            cudaMalloc((void **)&d_input, sizeof(ComplexType) * N * batch_size);
            cudaMalloc((void **)&d_output, sizeof(ComplexType) * N * batch_size);
            size_t ws = 0;

            // Copy the input data from host to device
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_input, h_input, sizeof(ComplexType) * N * batch_size, cudaMemcpyHostToDevice, stream));

            // Create a cuFFT plan
            cufftHandle plan;
            CUFFT_CHECK_AND_EXIT(cufftCreate(&plan));
#ifdef SINGLE_PRECISION
            CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan, N, CUFFT_C2C, batch_size));
#elif defined(DOUBLE_PRECISION)
            CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan, N, CUFFT_Z2Z, batch_size));
#elif defined(BFLOAT_PRECISION)
            CUFFT_CHECK_AND_EXIT(cufftXtMakePlanMany(plan, 1, &sig_size, NULL, 1, N, CUDA_C_16BF, NULL, 1, N, CUDA_C_16BF, batch_size, &ws, CUDA_C_16BF));
#else
            CUFFT_CHECK_AND_EXIT(cufftXtMakePlanMany(plan, 1, &sig_size, NULL, 1, N, CUDA_C_16F, NULL, 1, N, CUDA_C_16F, batch_size, &ws, CUDA_C_16F));
#endif
            cufftSetStream(plan, stream);

            // Measure the time taken for each FFT execution and sum the durations
            cudaEvent_t start, stop;
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&start));
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&stop));
            auto cufft_kernel = [&](cudaStream_t stream) {
#ifdef SINGLE_PRECISION
                CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD));
#elif defined(DOUBLE_PRECISION)
                CUFFT_CHECK_AND_EXIT(cufftExecZ2Z(plan, d_input, d_output, CUFFT_FORWARD));
#else
                cufftXtExec(plan, d_input, d_output, CUFFT_FORWARD);
#endif
            };
            float total_time = 0.0;
            nvmlDevice_t device;
            unsigned long long GPUEnergy1, GPUEnergy2, total_energy = 0.0;
            nvmlDeviceGetHandleByIndex_v2(0, &device);
            for (size_t i = 0; i < warmup_runs; i++) {
                cufft_kernel(stream);
            }
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
            nvmlDeviceGetTotalEnergyConsumption (device, &GPUEnergy1); 
            CUDA_CHECK_AND_EXIT(cudaEventRecord(start, stream));
            for (size_t i = 0; i < (num_runs_per_plan - warmup_runs); i++) {
                cufft_kernel(stream);
            }
            CUDA_CHECK_AND_EXIT(cudaEventRecord(stop, stream));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
            nvmlDeviceGetTotalEnergyConsumption (device, &GPUEnergy2);

            CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&total_time, start, stop));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(start));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(stop));
            total_energy = (GPUEnergy2 - GPUEnergy1);
            
            // Convert ms to s
            total_time = total_time / 1000.0;
            // Calculate the average duration and energy
            double avg_duration = total_time / (num_runs_per_plan - warmup_runs) / batch_size;
            double avg_energy = (double)(total_energy) / (num_runs_per_plan - warmup_runs) / batch_size;
            double power = ((double)(total_energy)*1.0E-3)/total_time;

            // Calculate GFLOPS
            double gflops = 5.0 * N * (std::log(N) / std::log(2)) / avg_duration / 1000000000.0;

            // Get Repeat Data
            total_gflops += gflops;
            total_avg_duration += avg_duration;
            total_power += power;
            total_avg_energy += avg_energy;
            // Output the total and average time taken for the FFT execution
            std::cout << "N = " << N << ", Repeat " << repeat + 1 << ": Total FFT execution time for " << num_runs_per_plan
                      << " runs: " << total_time << " seconds" << std::endl;
            std::cout << "Average FFT execution time per run: " << avg_duration << " seconds" << ", GFLOPS: " << gflops << std::endl;

            // Save the results to the CSV file
            // results_file << N << "," << batch_size << "," << num_runs_per_plan << "," << gflops << "," << avg_duration << "," << power << "\n";

            // Clean up
            CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));
            CUDA_CHECK_AND_EXIT(cudaFree(d_input));
            CUDA_CHECK_AND_EXIT(cudaFree(d_output));
            CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
            free(h_input);
        }
        // Average the accumulated values over the number of repeats
        double avg_gflops = total_gflops / num_repeats;
        double avg_duration_final = total_avg_duration / num_repeats;
        double power_final = total_power / num_repeats;  
        double avg_energy_final = total_avg_energy / num_repeats;
        
        // Save averaged parameters to file
        // Output the averaged results to the CSV file
        results_file << real_low << "," << real_high << "," << imag_low << "," << imag_high << ","
                     << N << "," << batch_size << "," << num_runs_per_plan << "," << num_repeats << "," << avg_gflops << "," << avg_duration_final << "," << power_final << "," << avg_energy_final << "\n";
    }

    // Close the CSV file
    results_file.close();

    // Shutdown NVML
    nvmlShutdown();

    return 0;
}
