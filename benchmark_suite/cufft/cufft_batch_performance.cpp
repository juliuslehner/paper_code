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
#include <cuda_fp16.h>
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

int main(int argc, char *argv[]) {
    // Different values of N (number of sample points) to test
    std::vector<int> N_values = {512};
    std::vector<int> Batch_values;
    int start_power = 1; // Starting power of 2 (2^1 = 2)
    int end_power = 20;  // Ending power of 2 (2^14 = 16384)

    // Number of runs to average the timing per plan execution
    int num_runs_per_plan = 5000;
    int warmup_runs = 10;

    // Get Arguments
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "s:e:r:w:")))
    {
        switch (opt_c)
        {
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
        case '?':
            printf("unkown option %c\n", optopt);
            break;
        default:
            break;
        }
    }
    Batch_values.push_back(1);
    for (int i = start_power; i <= end_power; ++i) {
        Batch_values.push_back(1<<i);
    }

    // Number of times to repeat the entire process for averaging
    const int num_repeats = 1;

    // CSV file to save the results
#ifdef SINGLE_PRECISION
    std::ofstream results_file("batch_testing/single_prec_benchmark_rand-10to10_upgrade.csv");
#elif defined(DOUBLE_PRECISION)
    std::ofstream results_file("batch_testing/double_prec_benchmark_rand-10to10_upgrade.csv");
#else 
    std::ofstream results_file("batch_testing/half_prec_benchmark_rand-10to10_upgrade.csv");
#endif

    results_file << "N,Batches,Num_Runs,GFLOPS,AverageTime(ms),AverageEnergy(mJ),AveragePower(W)\n";

    // Initialize NVML
    nvmlInit();

    // Create a CUDA stream
    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));


    for (int N : N_values) {
        for (int B : Batch_values) {
            long long sig_size = N;
            int batch_size = B;
            // Convert the FFT size to string
            std::string fftSizeStr = std::to_string(N);

            // Construct file paths
            // std::string inputFilePath = "precision_benchmarks_longdouble_extensive/" + fftSizeStr + "/fftw_double_rand_-10to10_input.csv";

            // Allocate host memory for the input array
            ComplexType *h_input = (ComplexType *)malloc(sizeof(ComplexType) * N * batch_size);
    
            // Initialize input array with a sample signal
            // std::vector<Complex> input = read_csv(inputFilePath);
            for (int i = 0; i < N; ++i) {
#ifdef HALF_PRECISION
            h_input[i].x = __double2half(((rand() / (double)RAND_MAX) * 20.0) - 10.0);
            h_input[i].y = __double2half(((rand() / (double)RAND_MAX) * 20.0) - 10.0);
#else
            h_input[i].x = static_cast<PrecType>(((rand() / (double)RAND_MAX) * 20.0) - 10.0);
            h_input[i].y = static_cast<PrecType>(((rand() / (double)RAND_MAX) * 20.0) - 10.0);
#endif
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
            
            // Calculate the average duration and energy
            double power = (double)(total_energy)/total_time;
            double avg_duration = total_time / (num_runs_per_plan - warmup_runs) / batch_size;
            double avg_energy = (double)(total_energy) / (num_runs_per_plan - warmup_runs) / batch_size;

            // Calculate GFLOPS
            double gflops = 5.0 * N * (std::log(N) / std::log(2)) / avg_duration / 1000000.0;


            // Output the total and average time taken for the FFT execution
            std::cout << "N = " << N << ": Total FFT execution time for " << num_runs_per_plan
                      << " runs: " << total_time << " milliseconds" << std::endl;
            std::cout << "Average FFT execution time per run: " << avg_duration << " milliseconds" << std::endl;

            // Save the results to the CSV file
            results_file << N << "," << batch_size << "," << num_runs_per_plan << "," << gflops << "," << avg_duration << "," << avg_energy << "," << power << "\n";

            // Clean up
            CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));
            CUDA_CHECK_AND_EXIT(cudaFree(d_input));
            CUDA_CHECK_AND_EXIT(cudaFree(d_output));
            free(h_input);
        }
    }

    // Close the CSV file
    results_file.close();

    // Destroy the CUDA stream
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    // Shutdown NVML
    nvmlShutdown();

    return 0;
}
