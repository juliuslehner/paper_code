//general parts
#include <stdio.h>
#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>

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
#include "utils_VkFFT.h"

VkFFTResult sample_0_benchmark_VkFFT_single(VkGPU* vkGPU, uint64_t file_output, FILE* output, double real_low, double real_high, double imag_low, double imag_high, uint64_t dimension, uint64_t isCompilerInitialized)
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
		fprintf(output, "re_low,re_high,im_low,im_high,Nx,Ny,Batches,Num_Runs,GFLOPS,AverageEnergy(mJ),AverageTime(s),AveragePower(W)\n");
	printf("0 - VkFFT FFT C2C benchmark 1D batched in single precision\n");
	const int num_runs = 3;
	// int max_exponent = 15;
	uint64_t (*benchmark_dimensions)[4];
	int num_benchmark_samples;
	if(dimension == 1){
		const int num_benchmark_samples_cond = 25;
		static uint64_t benchmark_dimensions_cond[num_benchmark_samples_cond][4]{
			{(uint64_t)pow(2,3), 1, 1, 1}, {(uint64_t)pow(2,4), 1, 1, 1}, {(uint64_t)pow(2,5), 1, 1, 1}, {(uint64_t)pow(2,6), 1, 1, 1},{(uint64_t)pow(2,7), 1, 1, 1},{(uint64_t)pow(2,8), 1, 1, 1},{(uint64_t)pow(2,9), 1, 1, 1},{(uint64_t)pow(2,10), 1, 1, 1},
			{(uint64_t)pow(2,11), 1, 1, 1},{(uint64_t)pow(2,12), 1, 1, 1},{(uint64_t)pow(2,13), 1, 1, 1},{(uint64_t)pow(2,14), 1, 1, 1},{(uint64_t)pow(2,15), 1, 1, 1},{(uint64_t)pow(2,16), 1, 1, 1},{(uint64_t)pow(2,17), 1, 1, 1},{(uint64_t)pow(2,18), 1, 1, 1},
			{(uint64_t)pow(2,19), 1, 1, 1},{(uint64_t)pow(2,20), 1, 1, 1},{(uint64_t)pow(2,21), 1, 1, 1},{(uint64_t)pow(2,22), 1, 1, 1},{(uint64_t)pow(2,23), 1, 1, 1},{(uint64_t)pow(2,24), 1, 1, 1},{(uint64_t)pow(2,25), 1, 1, 1},{(uint64_t)pow(2,26), 1, 1, 1},{(uint64_t)pow(2,27), 1, 1, 1}
		};
		benchmark_dimensions = benchmark_dimensions_cond;
		num_benchmark_samples = num_benchmark_samples_cond;
	} else{
		const int num_benchmark_samples_cond = 16;
		static uint64_t benchmark_dimensions_cond[num_benchmark_samples_cond][4] = {  
			{(uint64_t)pow(2,6), (uint64_t)pow(2,6), 1, 2}, {(uint64_t)pow(2,7), (uint64_t)pow(2,6), 1, 2}, {(uint64_t)pow(2,6), (uint64_t)pow(2,7), 1, 2},
			{(uint64_t)pow(2,7), (uint64_t)pow(2,7), 1, 2}, {(uint64_t)pow(2,8), (uint64_t)pow(2,7), 1, 2}, {(uint64_t)pow(2,7), (uint64_t)pow(2,8), 1, 2},
			{(uint64_t)pow(2,8), (uint64_t)pow(2,8), 1, 2}, {(uint64_t)pow(2,9), (uint64_t)pow(2,8), 1, 2}, {(uint64_t)pow(2,8), (uint64_t)pow(2,9), 1, 2},
			{(uint64_t)pow(2,9), (uint64_t)pow(2,9), 1, 2}, {(uint64_t)pow(2,10), (uint64_t)pow(2,9), 1, 2}, {(uint64_t)pow(2,9), (uint64_t)pow(2,10), 1, 2},
			{(uint64_t)pow(2,10), (uint64_t)pow(2,10), 1, 2}, {(uint64_t)pow(2,11), (uint64_t)pow(2,10), 1, 2}, {(uint64_t)pow(2,10), (uint64_t)pow(2,11), 1, 2},
			{(uint64_t)pow(2,11), (uint64_t)pow(2,11), 1, 2}
		};
		benchmark_dimensions = benchmark_dimensions_cond;
		num_benchmark_samples = num_benchmark_samples_cond;
	}
	// const int num_benchmark_samples = 25;
	// uint64_t benchmark_dimensions[num_benchmark_samples][4]{
	// 	{(uint64_t)pow(2,3), 1, 1, 1}, {(uint64_t)pow(2,4), 1, 1, 1}, {(uint64_t)pow(2,5), 1, 1, 1}, {(uint64_t)pow(2,6), 1, 1, 1},{(uint64_t)pow(2,7), 1, 1, 1},{(uint64_t)pow(2,8), 1, 1, 1},{(uint64_t)pow(2,9), 1, 1, 1},{(uint64_t)pow(2,10), 1, 1, 1},
	// 	{(uint64_t)pow(2,11), 1, 1, 1},{(uint64_t)pow(2,12), 1, 1, 1},{(uint64_t)pow(2,13), 1, 1, 1},{(uint64_t)pow(2,14), 1, 1, 1},{(uint64_t)pow(2,15), 1, 1, 1},{(uint64_t)pow(2,16), 1, 1, 1},{(uint64_t)pow(2,17), 1, 1, 1},{(uint64_t)pow(2,18), 1, 1, 1},
	// 	{(uint64_t)pow(2,19), 1, 1, 1},{(uint64_t)pow(2,20), 1, 1, 1},{(uint64_t)pow(2,21), 1, 1, 1},{(uint64_t)pow(2,22), 1, 1, 1},{(uint64_t)pow(2,23), 1, 1, 1},{(uint64_t)pow(2,24), 1, 1, 1},{(uint64_t)pow(2,25), 1, 1, 1},{(uint64_t)pow(2,26), 1, 1, 1},{(uint64_t)pow(2,27), 1, 1, 1}
	// };
	double benchmark_result = 0;//averaged result = sum(system_size/iteration_time)/num_benchmark_samples
	//memory allocated on the CPU once, makes benchmark completion faster + avoids performance issues connected to frequent allocation/deallocation.
	float* buffer_input = (float*)malloc((uint64_t)4 * 2 * (uint64_t)pow(2, (27)));
	if (!buffer_input) return VKFFT_ERROR_MALLOC_FAILED;
	// for (uint64_t i = 0; i < 2 * (uint64_t)pow(2, (27)); i++) {
	// 	buffer_input[i] = (float)(20 * ((float)rand()) / RAND_MAX - 10.0);
	// }
    // for (uint64_t k = 0; k < size_z; k++) {
    //     for (uint64_t j = 0; j < size_y; j++) {
    //         for (uint64_t i = 0; i < size_x; i++) {
    //             uint64_t index = i + j * size_x + k * (size_x * size_y);
    //             buffer_input[2 * index] = (static_cast<float>(rand()) / RAND_MAX) * (real_high - real_low) + real_low;       // Real part
    //             buffer_input[2 * index + 1] = (static_cast<float>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low;  // Imaginary part
    //         }
    //     }
    // }	
	for (uint64_t i = 0; i < (uint64_t)pow(2,27); ++i) {
		buffer_input[2 * i]     = static_cast<float>((static_cast<double>(rand()) / RAND_MAX) * (real_high - real_low) + real_low);  // Real part
		buffer_input[2 * i + 1] = static_cast<float>((static_cast<double>(rand()) / RAND_MAX) * (imag_high - imag_low) + imag_low);  // Imaginary part
	}
	// 1D
	// for (int N : N_values) {
	// 2D
	for (uint64_t n = 0; n < num_benchmark_samples; n++){
		// uint64_t n = (uint64_t)N;
		double run_time[num_runs];
		double run_energy[num_runs];
		for (uint64_t r = 0; r < num_runs; r++) {
			//Configuration + FFT application .
			VkFFTConfiguration configuration = {};
			VkFFTApplication app = {};
			//FFT + iFFT sample code.
			//Setting up FFT configuration for forward and inverse FFT.
			// 1D
			// configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).
			// configuration.size[0] = n; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			// if (n == 0) configuration.size[0] = 4096;
            // configuration.numberBatches = (uint64_t)((64 * 32 * (uint64_t)pow(2, (max_exponent - 9))) / configuration.size[0]);
			// if (configuration.numberBatches < 1) configuration.numberBatches = 1;
			// 2D
			configuration.FFTdim = benchmark_dimensions[n][3]; //FFT dimension, 1D, 2D or 3D (default 1).
			configuration.size[0] = benchmark_dimensions[n][0]; //Multidimensional FFT dimensions sizes (default 1). For best performance (and stability), order dimensions in descendant size order as: x>y>z.   
			configuration.size[1] = benchmark_dimensions[n][1];
			configuration.size[2] = benchmark_dimensions[n][2];
			configuration.numberBatches = (uint64_t)((64 * 32 * (uint64_t)pow(2, 16)) / (configuration.size[0]*configuration.size[1]));
#if(VKFFT_BACKEND!=5)
			if (r==0) configuration.saveApplicationToString = 1;
			if (r!=0) configuration.loadApplicationFromString = 1;
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
			//Allocate buffer for the input data.
			uint64_t bufferSize = (uint64_t)(sizeof(float) * 2 * configuration.size[0] * configuration.size[1] * configuration.size[2] * configuration.numberBatches);
#if(VKFFT_BACKEND==0)
			VkBuffer buffer = {};
			VkDeviceMemory bufferDeviceMemory = {};
			resFFT = allocateBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
			cuFloatComplex* buffer = 0;
			res = cudaMalloc((void**)&buffer, bufferSize);
			if (res != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==2)
			hipFloatComplex* buffer = 0;
			res = hipMalloc((void**)&buffer, bufferSize);
			if (res != hipSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = (void**)&buffer;
#elif(VKFFT_BACKEND==3)
			cl_mem buffer = 0;
			buffer = clCreateBuffer(vkGPU->context, CL_MEM_READ_WRITE, bufferSize, 0, &res);
			if (res != CL_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==4)
			void* buffer = 0;
			ze_device_mem_alloc_desc_t device_desc = {};
			device_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
			res = zeMemAllocDevice(vkGPU->context, &device_desc, bufferSize, sizeof(float), vkGPU->device, &buffer);
			if (res != ZE_RESULT_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
			configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==5)
            MTL::Buffer* buffer = 0;
            buffer = vkGPU->device->newBuffer(bufferSize, MTL::ResourceStorageModePrivate);
            configuration.buffer = &buffer;
#endif

			configuration.bufferSize = &bufferSize;

			//Fill data on CPU. It is best to perform all operations on GPU after initial upload.
			/*float* buffer_input = (float*)malloc(bufferSize);

			for (uint64_t k = 0; k < configuration.size[2]; k++) {
				for (uint64_t j = 0; j < configuration.size[1]; j++) {
					for (uint64_t i = 0; i < configuration.size[0]; i++) {
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1])] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						buffer_input[2 * (i + j * configuration.size[0] + k * (configuration.size[0]) * configuration.size[1]) + 1] = 2 * ((float)rand()) / RAND_MAX - 1.0;
						}
					}
				}

			*/
			//Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
            resFFT = transferDataFromCPU(vkGPU, buffer_input, &buffer, bufferSize);
            if (resFFT != VKFFT_SUCCESS) return resFFT;

			if (configuration.loadApplicationFromString) {
				FILE* kernelCache;
				uint64_t str_len;
				char fname[500];
				int VkFFT_version = VkFFTGetVersion();
				sprintf(fname, "VkFFT_binary");
				kernelCache = fopen(fname, "rb");
				if (!kernelCache) return VKFFT_ERROR_EMPTY_FILE;
				fseek(kernelCache, 0, SEEK_END);
				str_len = ftell(kernelCache);
				fseek(kernelCache, 0, SEEK_SET);
				configuration.loadApplicationString = malloc(str_len);
				fread(configuration.loadApplicationString, str_len, 1, kernelCache);
				fclose(kernelCache);
			}
			//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
			resFFT = initializeVkFFT(&app, configuration);
			if (resFFT != VKFFT_SUCCESS) return resFFT;

			if (configuration.loadApplicationFromString)
				free(configuration.loadApplicationString);

			if (configuration.saveApplicationToString) {
				FILE* kernelCache;
				char fname[500];
				int VkFFT_version = VkFFTGetVersion();
				sprintf(fname, "VkFFT_binary");
				kernelCache = fopen(fname, "wb");
				fwrite(app.saveApplicationString, app.applicationStringSize, 1, kernelCache);
				fclose(kernelCache);
			}

			//Submit FFT+iFFT.
			// uint64_t num_iter = (((uint64_t)3 * 4096 * 1024.0 * 1024.0) / bufferSize > 1000) ? 1000 : (uint64_t)(((uint64_t)3 * 4096 * 1024.0 * 1024.0) / bufferSize);
			uint64_t num_iter = 1000;
#if(VKFFT_BACKEND==0)
			if (vkGPU->physicalDeviceProperties.vendorID == 0x8086) num_iter /= 4;//smaller benchmark for Intel GPUs
#elif(VKFFT_BACKEND==3)
			cl_uint vendorID;
			clGetDeviceInfo(vkGPU->device, CL_DEVICE_VENDOR_ID, sizeof(cl_int), &vendorID, 0);
			if (vendorID == 0x8086) num_iter /= 4;//smaller benchmark for Intel GPUs
#elif(VKFFT_BACKEND==4)
			ze_device_properties_t device_properties;
			res = zeDeviceGetProperties(vkGPU->device, &device_properties);
			if (res != 0) return VKFFT_ERROR_FAILED_TO_GET_ATTRIBUTE;
			if (device_properties.vendorId == 0x8086) num_iter /= 4;//smaller benchmark for Intel GPUs
#endif
			if (num_iter == 0) num_iter = 1;
			double totTime = 0;
			double totEnergy = 0;
			VkFFTLaunchParams launchParams = {};
			resFFT = performVulkanFFT(vkGPU, &app, &launchParams, -1, num_iter, &totTime, &totEnergy);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
			// Get Energy and time per FFT
			run_time[r] = totTime;
			run_energy[r] = totEnergy;
			if (n > 0) {
				if (r == num_runs - 1) {
					// double std_error_time, std_error_energy = 0;
					double avg_time = 0.0;
					double avg_energy = 0.0;
					for (uint64_t t = 0; t < num_runs; t++) {
						avg_time += run_time[t];
						avg_energy += run_energy[t];
					}
					double power = avg_energy/avg_time;
					uint64_t total_ffts = num_iter*num_runs*configuration.numberBatches;
					// Get time per FFT
					avg_time = avg_time/total_ffts;
					avg_energy = avg_energy/total_ffts;
					// for (uint64_t t = 0; t < num_runs; t++) {
					// 	std_error_time += (run_time[t] - avg_time) * (run_time[t] - avg_time);
					// 	std_error_energy += (run_energy[t] - avg_energy) * (run_energy[t] - avg_energy);
					// }
					// std_error_time = sqrt(std_error_time / num_runs);
					// std_error_energy = sqrt(std_error_energy / num_runs);
					double gflops;
					if(configuration.size[1]  == 1){
						gflops = (5.0 * configuration.size[0] * configuration.size[1] * log2(configuration.size[0]*configuration.size[1]))/(avg_time * 1000000.0);
					} else{
						double flop = configuration.size[0] * 5.0 * std::log2(configuration.size[1]) * configuration.size[1] + configuration.size[1] * 5.0 * configuration.size[0] * std::log2(configuration.size[0]);;
						gflops = flop/(avg_time * 1000000.0);
					}
					// double power = avg_energy/avg_time;
					avg_time = avg_time * 1e-3; // get time in seconds
					uint64_t num_tot_transfers = 0;
					for (uint64_t i = 0; i < configuration.FFTdim; i++)
						num_tot_transfers += app.localFFTPlan->numAxisUploads[i];
					num_tot_transfers *= 4;
					if (file_output)
						fprintf(output, "%.2f,%.2f,%.2f,%.2f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.2f,%.4e,%.4e,%.4e\n", real_low, real_high, imag_low, imag_high, configuration.size[0], configuration.size[1], configuration.numberBatches, num_iter, gflops, avg_energy, avg_time, power);
					printf("VkFFT System: %" PRIu64 " %" PRIu64 "x%" PRIu64 " Buffer: %" PRIu64 " MB avg_time_per_fft: %0.8f s avg_power_per_fft: %0.8f W GFLOPS: %0.8f num_iter: %" PRIu64 " \n", (uint64_t)log2(configuration.size[0]), configuration.size[0], configuration.numberBatches, bufferSize / 1024 / 1024, avg_time, power, gflops, num_iter);
					benchmark_result += ((double)bufferSize / 1024) / avg_time;
				}


			}

#if(VKFFT_BACKEND==0)
			vkDestroyBuffer(vkGPU->device, buffer, NULL);
			vkFreeMemory(vkGPU->device, bufferDeviceMemory, NULL);
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

			deleteVkFFT(&app);

		}
	}
	free(buffer_input);
	benchmark_result /= 25;

	if (file_output) {
		// fprintf(output, "Benchmark score VkFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result));
#if(VKFFT_BACKEND==0)
		fprintf(output, "Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
		fclose(output);
	}
	printf("Benchmark score VkFFT: %" PRIu64 "\n", (uint64_t)(benchmark_result));
#if(VKFFT_BACKEND==0)
	printf("Device name: %s API:%d.%d.%d\n", vkGPU->physicalDeviceProperties.deviceName, (vkGPU->physicalDeviceProperties.apiVersion >> 22), ((vkGPU->physicalDeviceProperties.apiVersion >> 12) & 0x3ff), (vkGPU->physicalDeviceProperties.apiVersion & 0xfff));
#endif
	// printf("Return res\n");
	return resFFT;
}
