//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <iostream>
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
#ifdef USE_cuFFT
#include "precision_cuFFT_half.h"
#endif	
using half_float::half;

typedef half half2[2];

// struct Complex {
//     double real;
//     double imag;
// };

// void calculate_error(const std::vector<Complex>& data1, const std::vector<Complex>& data2, double *error_abs, double *error_rel) {
//     if (data1.size() != data2.size()) {
//         throw std::invalid_argument("Vectors must be of the same size.");
//     }
//     using namespace std;
//     double sum_diff_abs = 0.0;
//     double sum_diff_rel = 0.0;
// 	size_t num_entries = data1.size();
//     double real_diff, imag_diff, abs_error, magnitude;
//     for (size_t i = 0; i < data1.size(); ++i) {
//         real_diff = data1[i].real - data2[i].real;
//         imag_diff = data1[i].imag - data2[i].imag;
//         abs_error = abs(real_diff) + abs(imag_diff);
//         if (std::isnan(real_diff) || std::isnan(imag_diff) || std::isinf(real_diff) || std::isinf(imag_diff)) {
//             // std::cerr << "Warning: Detected NaN or Inf value in data at index " << i << std::endl;
// 			// std::cout << "Warning: Detected NaN or Inf value in data at index " << i << std::endl;
// 			num_entries--;
//             continue; // Skip this iteration
//         }
//         sum_diff_abs += abs_error;
//         // Get relative errors
//         magnitude = sqrt(data1[i].real * data1[i].real + data1[i].imag * data1[i].imag);
//         if (magnitude != 0) {
//             sum_diff_rel += abs_error / magnitude;
//         } else {
//             // Handle the case where magnitude is zero
//             sum_diff_rel += abs_error; // or another appropriate measure
//         }
//     }
//     if (num_entries > 0) {
//         *error_rel = sum_diff_rel / num_entries;
//         *error_abs = sum_diff_abs / num_entries;
//     } else {
//         *error_rel = -99.0;
//         *error_abs = -99.0;
// 		std::cout << "No valid entries!!" << std::endl;
//     }
// }

VkFFTResult sample_13_precision_VkFFT_half(VkGPU* vkGPU, uint64_t file_output, FILE* output, double real_low, double real_high, double imag_low, double imag_high, uint64_t dimension,uint64_t isCompilerInitialized)
{
	VkFFTResult resFFT = VKFFT_SUCCESS;
#if(VKFFT_BACKEND==0)
	VkResult res = VK_SUCCESS;
#elif(VKFFT_BACKEND==1)
	cudaError_t res = cudaSuccess;
#elif(VKFFT_BACKEND==2)
	hipError_t res = hipSuccess;
#elif(VKFFT_BACKEND==3)
	cl_int res = CL_SUCCESS;
#elif(VKFFT_BACKEND==4)
	ze_result_t res = ZE_RESULT_SUCCESS;
#elif(VKFFT_BACKEND==5)
#endif
	if (file_output)
		// fprintf(output, "re_low,re_high,im_low,im_high,Nx,Ny,Absolute_Error,Absolute_Error(%)\n");
		fprintf(output, "variance_real,variance_imag,Rate_of_Change,Magnitude,Nx,Ny,Absolute_Error,Absolute_Error(%)\n");
	printf("13 - VkFFT/FFTW C2C precision test in half precision\n");

	// const int num_benchmark_samples = 25;
	const int num_runs = 1;
	uint64_t (*benchmark_dimensions)[5];
	int num_benchmark_samples;
	if(dimension == 1){
		const int num_benchmark_samples_1d = 16;
		static uint64_t benchmark_dimensions_1d[num_benchmark_samples_1d][5]{
			// {(uint64_t)pow(2,23), 1, 1, 1, 1}
			{(uint64_t)pow(2,3), 1, 1, 1, 1}, {(uint64_t)pow(2,4), 1, 1, 1, 1}, {(uint64_t)pow(2,5), 1, 1, 1, 1}, {(uint64_t)pow(2,6), 1, 1, 1, 1},{(uint64_t)pow(2,7), 1, 1, 1, 1},{(uint64_t)pow(2,8), 1, 1, 1, 1},{(uint64_t)pow(2,9), 1, 1, 1, 1},{(uint64_t)pow(2,10), 1, 1, 1, 1},
			{(uint64_t)pow(2,11), 1, 1, 1, 1},{(uint64_t)pow(2,12), 1, 1, 1, 1},{(uint64_t)pow(2,13), 1, 1, 1, 1},{(uint64_t)pow(2,14), 1, 1, 1, 1},{(uint64_t)pow(2,15), 1, 1, 1, 1},{(uint64_t)pow(2,16), 1, 1, 1, 1},{(uint64_t)pow(2,17), 1, 1, 1, 1},{(uint64_t)pow(2,18), 1, 1, 1, 1}
			// {(uint64_t)pow(2,19), 1, 1, 1, 1},{(uint64_t)pow(2,20), 1, 1, 1, 1},{(uint64_t)pow(2,21), 1, 1, 1, 1},{(uint64_t)pow(2,22), 1, 1, 1, 1},{(uint64_t)pow(2,23), 1, 1, 1, 1},{(uint64_t)pow(2,24), 1, 1, 1, 1},{(uint64_t)pow(2,25), 1, 1, 1, 1},{(uint64_t)pow(2,26), 1, 1, 1, 1},{(uint64_t)pow(2,27), 1, 1, 1, 1}
		};
		benchmark_dimensions = benchmark_dimensions_1d;
		num_benchmark_samples = num_benchmark_samples_1d;
	} else{
		const int num_benchmark_samples_2d = 16;
		uint64_t benchmark_dimensions_2d[num_benchmark_samples_2d][5] = {  
			{(uint64_t)pow(2,6), (uint64_t)pow(2,6), 1, 1, 2}, {(uint64_t)pow(2,7), (uint64_t)pow(2,6), 1, 1, 2}, {(uint64_t)pow(2,6), (uint64_t)pow(2,7), 1, 1, 2},
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
	std::cout << "input arrays done" << std::endl;
	for (int n = 0; n < num_benchmark_samples; n++) {
		for (int r = 0; r < num_runs; r++) {

			half2* inputC;
			fftwl_complex* inputC_long_double;
			uint64_t dims[3] = { benchmark_dimensions[n][0] , benchmark_dimensions[n][1] ,benchmark_dimensions[n][2] };

			inputC = (half2*)(malloc(2 * sizeof(half) * dims[0] * dims[1] * dims[2]));
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
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][0] = (half)(real_part);
						inputC[i + j * dims[0] + l * dims[0] * dims[1]][1] = (half)(imag_part);
						inputC_long_double[i + j * dims[0] + l * dims[0] * dims[1]][0] = (long double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][0];
						inputC_long_double[i + j * dims[0] + l * dims[0] * dims[1]][1] = (long double)inputC[i + j * dims[0] + l * dims[0] * dims[1]][1];
						// std::cout << inputC_long_double[i + j * dims[0] + l * dims[0] * dims[1]][0] << ", "<< inputC_long_double[i + j * dims[0] + l * dims[0] * dims[1]][1] << std::endl;

						total_real_sum += real_part;
						total_imag_sum += imag_part;
						double magnitude = std::sqrt(real_part*real_part + imag_part*imag_part);
						total_magnitude_sum += magnitude;
						if (j > 0) {
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
			// std::cout << "FFTW Ouptut: " << std::endl;
			// size_t debug_count = 0;
			std::vector<Complex> data_highprecision(dims[0]*dims[1]);
			for (size_t i = 0; i < (dims[0]*dims[1]); ++i) {
				// debug_count++;
				data_highprecision[i].real = static_cast<double>(output_FFTW[i][0]);
				data_highprecision[i].imag = static_cast<double>(output_FFTW[i][1]);
				// if(debug_count > 1000){
				// 	std::cout << data_highprecision[i].real << ", " << data_highprecision[i].imag << std::endl;
				// 	debug_count = 0;
				// }
			}
			float totTime = 0;
			int num_iter = 1;

#ifdef USE_cuFFT
			half2* output_extFFT = (half2*)(malloc(2 * sizeof(half) * dims[0] * dims[1] * dims[2]));
			if (!output_extFFT) return VKFFT_ERROR_MALLOC_FAILED;
			launch_precision_cuFFT_half(inputC, (void*)output_extFFT, (int)vkGPU->device_id, benchmark_dimensions[n]);
#endif // USE_cuFFT

			//VkFFT part

			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			configuration.FFTdim = benchmark_dimensions[n][4]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];
			configuration.halfPrecision = true;

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
				bufferSize[i] = (uint64_t)sizeof(half) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] / numBuf;
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
				res = cudaMalloc((void**)&buffer, bufferSize[i]);
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
				res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize[i], sizeof(float), vkGPU->device, &buffer);
				if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
#elif(VKFFT_BACKEND==5)
                buffer = vkGPU->device->newBuffer(bufferSize[i], MTL::ResourceStorageModePrivate);
#endif
			}

			configuration.bufferNum = numBuf;
			/*
#if(VKFFT_BACKEND==0)
			configuration.buffer = buffer;
#elif(VKFFT_BACKEND==1)
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			configuration.buffer = (void**)&buffer;
#endif
			*/ // Can specify buffers at launch
			configuration.bufferSize = bufferSize;

			//Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
			uint64_t shift = 0;
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				resFFT = transferDataFromCPU(vkGPU, (inputC + shift / 2 / sizeof(half)), &buffer[i], bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#else
                resFFT = transferDataFromCPU(vkGPU, (inputC + shift / 2 / sizeof(half)), &buffer, bufferSize[i]);
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
			half2* output_VkFFT = (half2*)(malloc(2 * sizeof(half) * dims[0] * dims[1] * dims[2]));
			if (!output_VkFFT) return VKFFT_ERROR_MALLOC_FAILED;
			//Transfer data from GPU using staging buffer.
			shift = 0;
			for (uint64_t i = 0; i < numBuf; i++) {
#if(VKFFT_BACKEND==0)
				resFFT = transferDataToCPU(vkGPU, (output_VkFFT + shift / 2 / sizeof(half)), &buffer[i], bufferSize[i]);
				if (resFFT != VKFFT_SUCCESS) return resFFT;
#else
                resFFT = transferDataToCPU(vkGPU, (output_VkFFT + shift / 2 / sizeof(half)), &buffer, bufferSize[i]);
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

// 			double avg_difference[2] = { 0,0 };
// 			double max_difference[2] = { 0,0 };
// 			double avg_eps[2] = { 0,0 };
// 			double max_eps[2] = { 0,0 };
// 			for (uint64_t l = 0; l < dims[2]; l++) {
// 				for (uint64_t j = 0; j < dims[1]; j++) {
// 					for (uint64_t i = 0; i < dims[0]; i++) {
// 						uint64_t loc_i = i;
// 						uint64_t loc_j = j;
// 						uint64_t loc_l = l;

// 						//if (file_output) fprintf(output, "%.2e %.2e - %.2e %.2e \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
// 						//printf("%.2e %.2e - %.2e %.2e \n", output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] / N, output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] / N, output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][0], output_VkFFT[(loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1])][1]);
// 						double current_data_norm = sqrt(output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0] + output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1] * output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
// #ifdef USE_cuFFT
// 						double current_diff_x_extFFT = (output_extFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
// 						double current_diff_y_extFFT = (output_extFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
// 						double current_diff_norm_extFFT = sqrt(current_diff_x_extFFT * current_diff_x_extFFT + current_diff_y_extFFT * current_diff_y_extFFT);
// 						if (current_diff_norm_extFFT > max_difference[0]) max_difference[0] = current_diff_norm_extFFT;
// 						avg_difference[0] += current_diff_norm_extFFT;
// 						if ((current_diff_norm_extFFT / current_data_norm > max_eps[0])) {
// 							max_eps[0] = current_diff_norm_extFFT / current_data_norm;
// 						}
// 						avg_eps[0] += current_diff_norm_extFFT / current_data_norm;
// #endif

// 						double current_diff_x_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][0] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][0]);
// 						double current_diff_y_VkFFT = (output_VkFFT[loc_i + loc_j * dims[0] + loc_l * dims[0] * dims[1]][1] - output_FFTW[i + j * dims[0] + l * dims[0] * dims[1]][1]);
// 						double current_diff_norm_VkFFT = sqrt(current_diff_x_VkFFT * current_diff_x_VkFFT + current_diff_y_VkFFT * current_diff_y_VkFFT);
// 						if (current_diff_norm_VkFFT > max_difference[1]) max_difference[1] = current_diff_norm_VkFFT;
// 						avg_difference[1] += current_diff_norm_VkFFT;
// 						if ((current_diff_norm_VkFFT / current_data_norm > max_eps[1])) {
// 							max_eps[1] = current_diff_norm_VkFFT / current_data_norm;
// 						}
// 						avg_eps[1] += current_diff_norm_VkFFT / current_data_norm;
// 					}
// 				}
// 			}
// 			avg_difference[0] /= (dims[0] * dims[1] * dims[2]);
// 			avg_eps[0] /= (dims[0] * dims[1] * dims[2]);
// 			avg_difference[1] /= (dims[0] * dims[1] * dims[2]);
// 			avg_eps[1] /= (dims[0] * dims[1] * dims[2]);

#ifdef USE_cuFFT
			// if (file_output)
			// 	fprintf(output, "cuFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
			// printf("cuFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2], avg_difference[0], max_difference[0], avg_eps[0], max_eps[0]);
#endif
			if (file_output)
				// fprintf(output, "%.2f,%.2f,%.2f,%.2f,%" PRIu64 ",%" PRIu64 ",%.2e,%.2e\n", real_low, real_high, imag_low, imag_high, dims[0], dims[1], abs_error, rel_error*100);
				fprintf(output, "%.2f,%.2f,%.2f,%.2f,%" PRIu64 ",%" PRIu64 ",%.2e,%.2e\n", var_real, var_imag, rate_change, magnitude_mean, dims[0], dims[1], abs_error, rel_error*100);
			// printf("VkFFT System: %" PRIu64 "x%" PRIu64 "x%" PRIu64 " avg_difference: %.2e max_difference: %.2e avg_eps: %.2e max_eps: %.2e\n", dims[0], dims[1], dims[2], avg_difference[1], max_difference[1], avg_eps[1], max_eps[1]);
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
	fclose(output);
	return resFFT;
}
