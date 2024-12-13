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
#include "fftw3.h"
#include <unistd.h>

//general parts
#include <stdio.h>
#include <memory>
#include <chrono>
#include <thread>
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
#include "half.hpp"
#include "utils_VkFFT.h"
using half_float::half;
using namespace std;

typedef half half2[2];
extern char *optarg;
extern int optopt;

#ifdef SINGLE_PRECISION
typedef float PrecType;
#elif defined(DOUBLE_PRECISION)
typedef double PrecType;
#elif defined(HALF_PRECISION)
typedef half PrecType; 
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
void read_binary(const string& filename, PrecType* input, int n) {
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
        input[2*i] = (PrecType)(real);
        input[2*i + 1] = (PrecType)(imag);
    }
    file.close();
}

// Function to write to binary file
void write_binary(const std::string& filepath, PrecType* output, int n) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }

    // Write the real and imaginary parts interleaved
    for (int i = 0; i < n; ++i) {
        double real = static_cast<double>(output[2*i]);
        double imag = static_cast<double>(output[2*i + 1]);
        file.write(reinterpret_cast<const char*>(&real), sizeof(double));
        file.write(reinterpret_cast<const char*>(&imag), sizeof(double)); 
    }

    // Close the file
    file.close();
}

int main(int argc, char *argv[]) {
    int nx, ny;
    string input_filepath, output_filepath;
    string threads, signal_filename;
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

    //zero-initialize configuration + FFT application
    VkFFTConfiguration configuration = {};
    VkFFTApplication app = {};
    configuration.FFTdim = 2;
    configuration.size[0] = nx;
    configuration.size[1] = ny;
#ifdef HALF_PRECISION
    configuration.halfPrecision = true;
#elif defined(DOUBLE_PRECISION)
    configuration.doublePrecision = true;
#endif

    //Device management + code submission
#if(VKFFT_BACKEND==5)
    configuration.device = vkGPU->device;
#else
    configuration.device = &vkGPU->device;
#endif

#if(VKFFT_BACKEND==0) //Vulkan API
    configuration.queue = &vkGPU->queue;
    configuration.fence = &vkGPU->fence;
    configuration.commandPool = &vkGPU->commandPool;
    configuration.physicalDevice = &vkGPU->physicalDevice;
    configuration.isCompilerInitialized = isCompilerInitialized;
#elif(VKFFT_BACKEND==3) //OpenCL API
    configuration.context = &vkGPU->context;
#elif(VKFFT_BACKEND==4)
    configuration.context = &vkGPU->context;
    configuration.commandQueue = &vkGPU->commandQueue;
    configuration.commandQueueID = vkGPU->commandQueueID;
#elif(VKFFT_BACKEND==5)
    configuration.queue = vkGPU->queue;
#endif

    //Memory allocation on Host and input data creation
    uint64_t bufferSize = (uint64_t)sizeof(PrecType) * 2 * configuration.size[0] * configuration.size[1];
    configuration.bufferSize = &bufferSize;
    PrecType* buffer_cpu = (PrecType*)malloc(bufferSize);
    read_binary(input_filepath, buffer_cpu, nx*ny);
	if (!buffer_cpu) return VKFFT_ERROR_MALLOC_FAILED;

// Allocate Memory on GPU
#if(VKFFT_BACKEND==0)
    VkBuffer buffer = {};
    VkDeviceMemory bufferDeviceMemory = {};
    resFFT = allocateBuffer(vkGPU, &buffer, &bufferDeviceMemory, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSize);
    if (resFFT != VKFFT_SUCCESS) return resFFT;
    configuration.buffer = &buffer;
#elif(VKFFT_BACKEND==1)
    cuFloatComplex* buffer = 0;
    res2 = cudaMalloc((void**)&buffer, bufferSize);
    if (res2 != cudaSuccess) return VKFFT_ERROR_FAILED_TO_ALLOCATE;
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
    
    // Transfer Data to GPU
    resFFT = transferDataFromCPU(vkGPU, buffer_cpu, &buffer, bufferSize);
    if (resFFT != VKFFT_SUCCESS) return resFFT;    

    resFFT = initializeVkFFT(&app, configuration);
    VkFFTLaunchParams launchParams = {};
    launchParams.buffer = (void**)&buffer;
    #if(VKFFT_BACKEND==0) //Vulkan API
    launchParams.commandBuffer = &commandBuffer;
    #elif(VKFFT_BACKEND==3) //OpenCL API
    launchParams.commandQueue = &commandQueue;
    #elif(VKFFT_BACKEND==4) //Level Zero API
    launchParams->commandList = &commandList;
    #elif(VKFFT_BACKEND==5) //Metal API
    launchParams->commandBuffer = commandBuffer;
    launchParams->commandEncoder = commandEncoder;
    #endif
    resFFT = VkFFTAppend(&app, -1, &launchParams);

    // Copy the result from device to host
#if(VKFFT_BACKEND==0)
    resFFT = transferDataToCPU(vkGPU, buffer_cpu, &buffer, bufferSize);
    if (resFFT != VKFFT_SUCCESS) return resFFT;
#else
    resFFT = transferDataToCPU(vkGPU, buffer_cpu, &buffer, bufferSize);
    if (resFFT != VKFFT_SUCCESS) return resFFT;
#endif

    // Write results to the output file
    write_binary(output_filepath, buffer_cpu, nx*ny);

    cout << "Transform completed" << "\n";
    cout << "Results saved to " << output_filepath << "\n";

    // Clean up
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

#if(VKFFT_BACKEND==0)
    free(buffer);
    free(bufferDeviceMemory);
#endif
    // free(bufferSize);
    deleteVkFFT(&app);
    free(buffer_cpu);
	return resFFT;
}

