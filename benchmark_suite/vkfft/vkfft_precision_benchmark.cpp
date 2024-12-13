//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <thread>
#include <iomanip>
#include <unistd.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <calculate_error.h>

#if(VKFFT_BACKEND==0)
#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
#elif(VKFFT_BACKEND==1)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#elif(VKFFT_BACKEND==2)
#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>
#elif(VKFFT_BACKEND==3)
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif 
#elif(VKFFT_BACKEND==4)
#include <ze_api.h>
#elif(VKFFT_BACKEND==5)
#include "Foundation/Foundation.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "Metal/Metal.hpp"
#endif
#include "vkFFT.h"
#include "half.hpp"
#include "utils_VkFFT.h"
#include "fftw3.h"
using half_float::half;
typedef half half2[2];

#ifdef SINGLE_PRECISION
typedef float PrecType;
typedef fftwf_complex ComplexType;
#elif defined(DOUBLE_PRECISION)
typedef double PrecType;
typedef fftw_complex ComplexType;
#elif defined(HALF_PRECISION)
typedef half PrecType; 
typedef half2 ComplexType;
#else
#error "You must define one of SINGLE_PRECISION, DOUBLE_PRECISION or HALF_PRECISION"
#endif

// struct Complex {
//     double real;
//     double imag;
// };

std::string double_to_string(double value, int precision = 1){
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    std::string result = ss.str();
    std::replace(result.begin(), result.end(), '.', '_');
    return result;
}

void calculate_error(const std::vector<Complex>& data1, const std::vector<Complex>& data2, double *error_abs, double *error_rel) {
    if (data1.size() != data2.size()) {
        throw std::invalid_argument("Vectors must be of the same size.");
    }
    using namespace std;
    double sum_diff_abs = 0.0;
    double sum_diff_rel = 0.0;
	size_t num_entries = data1.size();
    double real_diff, imag_diff, abs_error, magnitude;
    for (size_t i = 0; i < data1.size(); ++i) {
        real_diff = data1[i].real - data2[i].real;
        imag_diff = data1[i].imag - data2[i].imag;
        abs_error = abs(real_diff) + abs(imag_diff);
        if (std::isnan(real_diff) || std::isnan(imag_diff) || std::isinf(real_diff) || std::isinf(imag_diff)) {
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
		std::cout << "No valid entries!!" << std::endl;
    }
}

int main(int argc, char *argv[]) {
    std::string output_filepath, precision;
    double real_low, real_high, imag_low, imag_high;
    int dimension;
    // Get Arguments
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "o:d:1:2:3:4:")))
    {
        switch (opt_c)
        {
        case 'o':
            output_filepath = strdup(optarg);
            break;
        case 'd':
            dimension = atoi(optarg);
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

// GPU Initializations
    VkFFTResult resFFT = VKFFT_SUCCESS;
    VkGPU vkGPU_base = {};
    VkGPU* vkGPU = &vkGPU_base;
    vkGPU->device_id = 0;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
	//create instance - a connection between the application and the Vulkan library 
	res = createInstance(vkGPU, sample_id);
	if (res != 0) {
		//printf("Instance creation failed, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE;
	}
	//set up the debugging messenger 
	res = setupDebugMessenger(vkGPU);
	if (res != 0) {
		//printf("Debug messenger creation failed, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER;
	}
	//check if there are GPUs that support Vulkan and select one
	res = findPhysicalDevice(vkGPU);
	if (res != 0) {
		//printf("Physical device not found, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE;
	}
	//create logical device representation
	res = createDevice(vkGPU, sample_id);
	if (res != 0) {
		//printf("Device creation failed, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_CREATE_DEVICE;
	}
	//create fence for synchronization 
	res = createFence(vkGPU);
	if (res != 0) {
		//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_CREATE_FENCE;
	}
	//create a place, command buffer memory is allocated from
	res = createCommandPool(vkGPU);
	if (res != 0) {
		//printf("Fence creation failed, error code: %" PRIu64 "\n", res);
		return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL;
	}
	vkGetPhysicalDeviceProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceProperties);
	vkGetPhysicalDeviceMemoryProperties(vkGPU->physicalDevice, &vkGPU->physicalDeviceMemoryProperties);

	glslang_initialize_process();//compiler can be initialized before VkFFT
#elif(VKFFT_BACKEND==1)
	CUresult res = CUDA_SUCCESS;
	cudaError_t res2 = cudaSuccess;
	res = cuInit(0);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	res2 = cudaSetDevice((int)vkGPU->device_id);
	if (res2 != cudaSuccess) return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
	res = cuDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
	res = cuCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
	if (res != CUDA_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
	res = hipInit(0);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	res = hipSetDevice((int)vkGPU->device_id);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID;
	res = hipDeviceGet(&vkGPU->device, (int)vkGPU->device_id);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
	res = hipCtxCreate(&vkGPU->context, 0, (int)vkGPU->device);
	if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
	cl_uint numPlatforms;
	res = clGetPlatformIDs(0, 0, &numPlatforms);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
	if (!platforms) return VKFFT_ERROR_MALLOC_FAILED;
	res = clGetPlatformIDs(numPlatforms, platforms, 0);
	if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	uint64_t k = 0;
	for (uint64_t j = 0; j < numPlatforms; j++) {
		cl_uint numDevices;
		res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, 0, 0, &numDevices);
		cl_device_id* deviceList = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
		if (!deviceList) return VKFFT_ERROR_MALLOC_FAILED;
		res = clGetDeviceIDs(platforms[j], CL_DEVICE_TYPE_ALL, numDevices, deviceList, 0);
		if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
		for (uint64_t i = 0; i < numDevices; i++) {
			if (k == vkGPU->device_id) {
				vkGPU->platform = platforms[j];
				vkGPU->device = deviceList[i];
				vkGPU->context = clCreateContext(NULL, 1, &vkGPU->device, NULL, NULL, &res);
				if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;
				cl_command_queue commandQueue = clCreateCommandQueue(vkGPU->context, vkGPU->device, 0, &res);
				if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
				vkGPU->commandQueue = commandQueue;
				i=numDevices;
				j=numPlatforms;
			}
			else {
				k++;
			}
		}
		free(deviceList);
	}
	free(platforms);
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
	res = zeInit(0);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	uint32_t numDrivers = 0;
	res = zeDriverGet(&numDrivers, 0);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	ze_driver_handle_t* drivers = (ze_driver_handle_t*)malloc(numDrivers * sizeof(ze_driver_handle_t));
	if (!drivers) return VKFFT_ERROR_MALLOC_FAILED;
	res = zeDriverGet(&numDrivers, drivers);
	if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_INITIALIZE;
	uint64_t k = 0;
	for (uint64_t j = 0; j < numDrivers; j++) {
		uint32_t numDevices = 0;
		res = zeDeviceGet(drivers[j], &numDevices, nullptr);
		if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
		ze_device_handle_t* deviceList = (ze_device_handle_t*)malloc(numDevices * sizeof(ze_device_handle_t));
		if (!deviceList) return VKFFT_ERROR_MALLOC_FAILED;
		res = zeDeviceGet(drivers[j], &numDevices, deviceList);
		if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_GET_DEVICE;
		for (uint64_t i = 0; i < numDevices; i++) {
			if (k == vkGPU->device_id) {
				vkGPU->driver = drivers[j];
				vkGPU->device = deviceList[i];
				ze_context_desc_t contextDescription = {};
				contextDescription.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
				res = zeContextCreate(vkGPU->driver, &contextDescription, &vkGPU->context);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT;

				uint32_t queueGroupCount = 0;
				res = zeDeviceGetCommandQueueGroupProperties(vkGPU->device, &queueGroupCount, 0);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;

				ze_command_queue_group_properties_t* cmdqueueGroupProperties = (ze_command_queue_group_properties_t*)malloc(queueGroupCount * sizeof(ze_command_queue_group_properties_t));
				if (!cmdqueueGroupProperties) return VKFFT_ERROR_MALLOC_FAILED;
				res = zeDeviceGetCommandQueueGroupProperties(vkGPU->device, &queueGroupCount, cmdqueueGroupProperties);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;

				uint32_t commandQueueID = -1;
				for (uint32_t i = 0; i < queueGroupCount; ++i) {
					if ((cmdqueueGroupProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) && (cmdqueueGroupProperties[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY)) {
						commandQueueID = i;
						break;
					}
				}
				if (commandQueueID == -1) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
				vkGPU->commandQueueID = commandQueueID;
				ze_command_queue_desc_t commandQueueDescription = {};
				commandQueueDescription.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
				commandQueueDescription.ordinal = commandQueueID;
				commandQueueDescription.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
				commandQueueDescription.mode = ZE_COMMAND_QUEUE_MODE_DEFAULT;
				res = zeCommandQueueCreate(vkGPU->context, vkGPU->device, &commandQueueDescription, &vkGPU->commandQueue);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE;
				free(cmdqueueGroupProperties);
				i=numDevices;
				j=numDrivers;
			}
			else {
				k++;
			}
		}

		free(deviceList);
	}
	free(drivers);
#elif(VKFFT_BACKEND==5)
    NS::Array* devices = MTL::CopyAllDevices();
    MTL::Device* device = (MTL::Device*)devices->object(vkGPU->device_id);
    vkGPU->device = device;
    MTL::CommandQueue* queue = device->newCommandQueue();
    vkGPU->queue = queue;
#endif
	uint64_t isCompilerInitialized = 1;

#ifdef SINGLE_PRECISION
    precision = "single";
#elif defined(DOUBLE_PRECISION)
    precision = "double";
#else 
    precision = "half";
#endif

    // Create output file
    std::stringstream filename;
    filename << output_filepath
             << precision << "_prec_benchmark_imag" << double_to_string(imag_low,0) << "to" << double_to_string(imag_high,0) << "_real" << double_to_string(real_low,0) << "to" << double_to_string(real_high,0) << ".csv";
    std::string filename_str = filename.str();
    std::ofstream results_file(filename_str);
    results_file << "variance_real,variance_imag,Rate_of_Change, Magnitude, Nx, Ny, Absolute_Error, Relative_Error(%)\n";

	// const int num_benchmark_samples = 25;
	const int num_runs = 1;
	uint64_t (*benchmark_dimensions)[5];
	int num_benchmark_samples;
	if(dimension == 1){
		const int num_benchmark_samples_1d = 25;
		static uint64_t benchmark_dimensions_1d[num_benchmark_samples_1d][5]{
			{(uint64_t)pow(2,3), 1, 1, 1, 1}, {(uint64_t)pow(2,4), 1, 1, 1, 1}, {(uint64_t)pow(2,5), 1, 1, 1, 1}, {(uint64_t)pow(2,6), 1, 1, 1, 1},{(uint64_t)pow(2,7), 1, 1, 1, 1},{(uint64_t)pow(2,8), 1, 1, 1, 1},{(uint64_t)pow(2,9), 1, 1, 1, 1},{(uint64_t)pow(2,10), 1, 1, 1, 1},
			{(uint64_t)pow(2,11), 1, 1, 1, 1},{(uint64_t)pow(2,12), 1, 1, 1, 1},{(uint64_t)pow(2,13), 1, 1, 1, 1},{(uint64_t)pow(2,14), 1, 1, 1, 1},{(uint64_t)pow(2,15), 1, 1, 1, 1},{(uint64_t)pow(2,16), 1, 1, 1, 1},{(uint64_t)pow(2,17), 1, 1, 1, 1},{(uint64_t)pow(2,18), 1, 1, 1, 1},
			{(uint64_t)pow(2,19), 1, 1, 1, 1},{(uint64_t)pow(2,20), 1, 1, 1, 1},{(uint64_t)pow(2,21), 1, 1, 1, 1},{(uint64_t)pow(2,22), 1, 1, 1, 1},{(uint64_t)pow(2,23), 1, 1, 1, 1},{(uint64_t)pow(2,24), 1, 1, 1, 1},{(uint64_t)pow(2,25), 1, 1, 1, 1},{(uint64_t)pow(2,26), 1, 1, 1, 1},{(uint64_t)pow(2,27), 1, 1, 1, 1}
		};
		benchmark_dimensions = benchmark_dimensions_1d;
		num_benchmark_samples = num_benchmark_samples_1d;
	} else{
		const int num_benchmark_samples_2d = 13;
		static uint64_t benchmark_dimensions_2d[num_benchmark_samples_2d][5] = {  
			{(uint64_t)pow(2,7), (uint64_t)pow(2,7), 1, 1, 2}, {(uint64_t)pow(2,8), (uint64_t)pow(2,7), 1, 1, 2}, {(uint64_t)pow(2,7), (uint64_t)pow(2,8), 1, 1, 2},
			{(uint64_t)pow(2,8), (uint64_t)pow(2,8), 1, 1, 2}, {(uint64_t)pow(2,9), (uint64_t)pow(2,8), 1, 1, 2}, {(uint64_t)pow(2,8), (uint64_t)pow(2,9), 1, 1, 2},
			{(uint64_t)pow(2,9), (uint64_t)pow(2,9), 1, 1, 2}, {(uint64_t)pow(2,10), (uint64_t)pow(2,9), 1, 1, 2}, {(uint64_t)pow(2,9), (uint64_t)pow(2,10), 1, 1, 2},
			{(uint64_t)pow(2,10), (uint64_t)pow(2,10), 1, 1, 2}, {(uint64_t)pow(2,11), (uint64_t)pow(2,10), 1, 1, 2}, {(uint64_t)pow(2,10), (uint64_t)pow(2,11), 1, 1, 2},
			{(uint64_t)pow(2,11), (uint64_t)pow(2,11), 1, 1, 2}
		};
		benchmark_dimensions = benchmark_dimensions_2d;
		num_benchmark_samples = num_benchmark_samples_2d;
	}

	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	for (int n = 0; n < num_benchmark_samples; n++) {
		for (int r = 0; r < num_runs; r++) {

			ComplexType* inputC;
			fftwl_complex* inputC_long_double;
			uint64_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			inputC = (ComplexType*)(malloc(2 * sizeof(PrecType) * dims[0] * dims[1] * dims[2]));
			if (!inputC) return VKFFT_ERROR_MALLOC_FAILED;
			inputC_long_double = (fftwl_complex*)(malloc(sizeof(fftwl_complex) * dims[0] * dims[1] * dims[2]));
			if (!inputC_long_double) return VKFFT_ERROR_MALLOC_FAILED;

			double real_part, imag_part, prev_real, prev_imag;
		    double total_real_sum = 0.0, total_imag_sum = 0.0;
			double total_magnitude_sum = 0.0, total_rate_change_sum = 0.0;
			double total_variance_real = 0.0, total_variance_imag = 0.0;
			// printf("FFTW content: \n");
			for (uint64_t l = 0; l < dims[2]; l++) {
				for (uint64_t j = 0; j < dims[1]; j++) {
					for (uint64_t i = 0; i < dims[0]; i++) {
						real_part = (static_cast<double>(rand()) / RAND_MAX) * (real_high - real_low) + real_low;
						imag_part = (static_cast<double>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low;
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][0] = (PrecType)(real_part);
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][1] = (PrecType)(imag_part);
						inputC_long_double[i + j * dims[0] + l * dims[0] * dims[1]][0] = (long double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][0];
						inputC_long_double[i + j * dims[0] + l * dims[0] * dims[1]][1] = (long double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][1];

						total_real_sum += real_part;
						total_imag_sum += imag_part;
						double magnitude = std::sqrt(real_part*real_part + imag_part*imag_part);
						total_magnitude_sum += magnitude;
						if (i > 0) {
							prev_real = (double)(inputC[i - 1 + j * dims[0] + l * dims[0] * dims[1]][0]);
							prev_imag = (double)(inputC[i - 1 + j * dims[0] + l * dims[0] * dims[1]][1]);
							double prev_magnitude = std::sqrt(
								prev_real * prev_real + 
								prev_imag * prev_imag
							);
							total_rate_change_sum += std::abs(magnitude - prev_magnitude);
						}
					}
				}
			}

			double overall_real_mean = total_real_sum/(dims[0]*dims[1]*dims[2]);
			double overall_imag_mean = total_imag_sum/(dims[0]*dims[1]*dims[2]);
			for (uint64_t l = 0; l < dims[2]; l++) {
				for (uint64_t j = 0; j < dims[1]; j++) {
					for (uint64_t i = 0; i < dims[0]; i++) {
			            total_variance_real += std::pow((double)(inputC[i + j * dims[0] + l * dims[0] * dims[1]][0]) - overall_real_mean, 2);
            			total_variance_imag += std::pow((double)(inputC[i + j * dims[0] + l * dims[0] * dims[1]][1]) - overall_imag_mean, 2);
					}
				}
			}

			double magnitude_mean = total_magnitude_sum / (dims[0]*dims[1]*dims[2]);
			double rate_change = total_rate_change_sum / (dims[0]*dims[1]*dims[2]);
			double var_real = total_variance_real / (dims[0]*dims[1]*dims[2]);
			double var_imag = total_variance_imag / (dims[0]*dims[1]*dims[2]);

			fftwl_plan p;

			fftwl_complex* output_FFTW = (fftwl_complex*)(malloc(sizeof(fftwl_complex) * dims[0] * dims[1] * dims[2]));
			if (!output_FFTW) return VKFFT_ERROR_MALLOC_FAILED;
			switch (benchmark_dimensions[n][4]) {
			case 1:
				p = fftwl_plan_dft_1d((int)benchmark_dimensions[n][0], inputC_long_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 2:
				p = fftwl_plan_dft_2d((int)benchmark_dimensions[n][1], (int)benchmark_dimensions[n][0], inputC_long_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			case 3:
				p = fftwl_plan_dft_3d((int)benchmark_dimensions[n][2], (int)benchmark_dimensions[n][1], (int)benchmark_dimensions[n][0], inputC_long_double, output_FFTW, -1, FFTW_ESTIMATE);
				break;
			}
		
			fftwl_execute(p);

			// Read in output vector
			std::vector<Complex> data_highprecision(dims[0]*dims[1]);
			for (size_t i = 0; i < (dims[0]*dims[1]); ++i) {
				data_highprecision[i].real = static_cast<double>(output_FFTW[i][0]);
				data_highprecision[i].imag = static_cast<double>(output_FFTW[i][1]);
			}
			float totTime = 0;
			int num_iter = 1;

			//VkFFT Configuration
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			configuration.FFTdim = benchmark_dimensions[n][4]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];
#ifdef HALF_PRECISION
            configuration.halfPrecision = true;
#elif defined(DOUBLE_PRECISION)
            configuration.doublePrecision = true;
#endif

			//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.
#if(VKFFT_BACKEND==5)
            configuration.device = vkGPU->device;
#else
            configuration.device = &vkGPU->device;
#endif
#if(VKFFT_BACKEND==0)
			configuration.queue = &vkGPU->queue; //to allocate memory for LUT, we have to pass a queue, vkGPU->fence, commandPool and physicalDevice pointers 
			configuration.fence = &vkGPU->fence;
			configuration.commandPool = &vkGPU->commandPool;
			configuration.physicalDevice = &vkGPU->physicalDevice;
			configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization
#elif(VKFFT_BACKEND==3)
			configuration.context = &vkGPU->context;
#elif(VKFFT_BACKEND==4)
			configuration.context = &vkGPU->context;
			configuration.commandQueue = &vkGPU->commandQueue;
			configuration.commandQueueID = vkGPU->commandQueueID;
#elif(VKFFT_BACKEND==5)
            configuration.queue = vkGPU->queue;
#endif

			uint64_t numBuf = 1;

			//Allocate buffers for the input data. - we use 4 in this example
			uint64_t* bufferSize = (uint64_t*)malloc(sizeof(uint64_t) * numBuf);
			if (!bufferSize) return VKFFT_ERROR_MALLOC_FAILED;
			for (uint64_t i = 0; i < numBuf; i++) {
				bufferSize[i] = {};
				bufferSize[i] = (uint64_t)sizeof(PrecType) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] / numBuf;
			}
#if(VKFFT_BACKEND==0)
			VkBuffer* buffer = (VkBuffer*)malloc(numBuf * sizeof(VkBuffer));
			if (!buffer) return VKFFT_ERROR_MALLOC_FAILED;
			VkDeviceMemory* bufferDeviceMemory = (VkDeviceMemory*)malloc(numBuf * sizeof(VkDeviceMemory));
			if (!bufferDeviceMemory) return VKFFT_ERROR_MALLOC_FAILED;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
#elif(VKFFT_BACKEND==3)
			cl_mem buffer = 0;
#elif(VKFFT_BACKEND==4)
			void* buffer = 0;
#elif(VKFFT_BACKEND==5)
            MTL::Buffer* buffer = 0;
#endif			
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				buffer[i] = {};
				bufferDeviceMemory[i] = {};
				resFFT = allocateBuffer(vkGPU, &buffer[i], &bufferDeviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#elif(VKFFT_BACKEND==1)
				res2 = cudaMalloc((void**)&buffer, bufferSize[i]);
				if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==2)
				res = hipMalloc((void**)&buffer, bufferSize[i]);
				if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==3)
				buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize[i], 0, &res);
				if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==4)
				ze_device_mem_alloc_desc_t device_desc = {};
				device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
				res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize[i], sizeof(PrecType), vkGPU->device, &buffer);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==5)
                buffer = vkGPU->device->newBuffer(bufferSize[i], MTL::ResourceStorageModePrivate);
#endif
			}
            // Can specify buffers at launch
			configuration.bufferNum = numBuf;
			configuration.bufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				resFFT = transferDataFromCPU(vkGPU, (inputC + shift / 2 / sizeof(PrecType)), &buffer[i], bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#else
                resFFT = transferDataFromCPU(vkGPU, (inputC + shift / 2 / sizeof(PrecType)), &buffer, bufferSize[i]);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
#endif
				shift += bufferSize[i];
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			resFFT = initializeVkFFT(&app, configuration);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			//Submit FFT+iFFT.
			//num_iter = 1;
			//specify buffers at launch
			VkFFTLaunchParams launchParams = {};
#if(VKFFT_BACKEND==0)
			launchParams.buffer = buffer;
#elif(VKFFT_BACKEND==1)
			launchParams.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			launchParams.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==3)
			launchParams.buffer = &buffer;
#elif(VKFFT_BACKEND==4)
			launchParams.buffer = (void**)&buffer;
#else
            launchParams.buffer = &buffer;
#endif
			resFFT = performVulkanFFT(vkGPU, &app, &launchParams, -1, num_iter);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			ComplexType* output_VkFFT = (ComplexType*)(malloc(2 * sizeof(PrecType) * dims[0] * dims[1] * dims[2]));
			if (!output_VkFFT) return VKFFT_ERROR_MALLOC_FAILED;
			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				resFFT = transferDataToCPU(vkGPU, (output_VkFFT + shift / 2 / sizeof(PrecType)), &buffer[i], bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#else
                resFFT = transferDataToCPU(vkGPU, (output_VkFFT + shift / 2 / sizeof(PrecType)), &buffer, bufferSize[i]);
                if (resFFT != VKFFT_SUCCESS) return resFFT;
#endif
				shift += bufferSize[i];
			}
			// std::cout << "VkFFT Output: " << std::endl;
			// size_t debug_count_1 = 0;
			std::vector<Complex> data_lowprecision(dims[0] * dims[1] * dims[2]);
			for (uint64_t l = 0; l < dims[2]; l++) {
				for (uint64_t j = 0; j < dims[1]; j++) {
					for (uint64_t i = 0; i < dims[0]; i++) {
						uint64_t index = i + j * dims[0] + l * dims[0] * dims[1];
						data_lowprecision[index].real = static_cast<double>(output_VkFFT[index][0]);
						data_lowprecision[index].imag = static_cast<double>(output_VkFFT[index][1]);
						// if(debug_count_1 > 1000){
						// 	std::cout << data_highprecision[i].real << ", " << data_highprecision[i].imag << std::endl;
						// 	debug_count_1 = 0;
						// }
					}
				}
			}
			double rel_error,abs_error = 0.0;
			calculate_error(data_lowprecision, data_highprecision, &abs_error, &rel_error);
			results_file << var_real << "," << var_imag << "," << rate_change << "," << magnitude_mean << "," << dims[0] << "," << dims[1] << "," << abs_error << "," << rel_error*100 << "\n";
			printf("VkFFT System: %" PRIu64 "x%" PRIu64 " abs_error: %.2e rel_error: %.2e\n", dims[0], dims[1], abs_error, rel_error*100);
			free(output_VkFFT);
			for (uint64_t i = 0; i < numBuf; i++) {

#if(VKFFT_BACKEND==0)
				vkDestroyBuffer(vkGPU->device, buffer[i], NULL);
				vkFreeMemory(vkGPU->device, bufferDeviceMemory[i], NULL);
#elif(VKFFT_BACKEND==1)
				cudaFree(buffer);
#elif(VKFFT_BACKEND==2)
				hipFree(buffer);
#elif(VKFFT_BACKEND==3)
				clReleaseMemObject(buffer);
#elif(VKFFT_BACKEND==4)
				zeMemFree(vkGPU->context, buffer);
#elif(VKFFT_BACKEND==5)
                buffer->release();
#endif

			}
#if(VKFFT_BACKEND==0)
			free(buffer);
			free(bufferDeviceMemory);
#endif
#ifdef USE_cuFFT
			free(output_extFFT);
#endif
			free(bufferSize);
			deleteVkFFT(&app);
			free(inputC);
			fftwl_destroy_plan(p);
			free(inputC_long_double);
			free(output_FFTW);
		}
	}
	return resFFT;
}
