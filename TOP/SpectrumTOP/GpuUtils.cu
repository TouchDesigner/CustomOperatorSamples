/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/
#include "GpuUtils.cuh"

#include "cuda_runtime.h"
#include "cuda_fp16.h"

#include <opencv2/core/cuda.hpp>

namespace
{
	void 
	createSurfaceObj(cudaSurfaceObject_t* sObj, cudaArray* arr)
	{
		cudaResourceDesc resDesc = {};
		resDesc.res.array.array = arr;
		resDesc.resType = cudaResourceTypeArray;
		cudaCreateSurfaceObject(sObj, &resDesc);
	}

	__global__ void
	copySurfaceToMat(cudaSurfaceObject_t src,
		uchar* dst, size_t dstStep,
		int width, int height)
	{
		// Calculate surface coordinates
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < width && y < height) {
			uchar data;
			// Read from input surface
			surf2Dread(&data, src, x, y);
			// Write to output mat data
			uchar* pixel = dst + (height - y - 1) * dstStep + x;
			*pixel = data;
		}
	}

	// Assume src is in 32F and dst is 32FC2
	__global__ void
	copy32FSurfaceToComplexMat(cudaSurfaceObject_t src,
		uchar* dst, size_t dstStep,
		size_t width, size_t height, size_t xOffset, size_t pixelSize)
	{
		// Calculate surface coordinates
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < width && y < height) {
			float2 data{};
			// Read from input surface
			surf2Dread(&data.x, src, x * pixelSize + xOffset, y);
			// Write to output mat data
			float2* pixel = reinterpret_cast<float2*>(dst + (height - y - 1) * dstStep + x * 2 * sizeof(float));
			*pixel = data;
		}
	}

	// Assume src is in 16F and dst is 32FC2
	__global__ void
	copy16FSurfaceToComplexMat(cudaSurfaceObject_t src,
		uchar* dst, size_t dstStep,
		size_t width, size_t height, size_t xOffset, size_t pixelSize)
	{
		// Calculate surface coordinates
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < width && y < height) {
			float2 data{};
			// Read from input surface
			half x16bits;
			surf2Dread(reinterpret_cast<uint16_t*>(&x16bits), src, x * pixelSize + xOffset, y);
			data.x = x16bits;
			// Write to output mat data
			float2* pixel = reinterpret_cast<float2*>(dst + (height - y - 1) * dstStep + x * 2 * sizeof(float));
			*pixel = data;
		}
	}

	// Assume src is in 16U and dst is 32FC2
	__global__ void
	copy16USurfaceToComplexMat(cudaSurfaceObject_t src,
		uchar* dst, size_t dstStep,
		size_t width, size_t height, size_t xOffset, size_t pixelSize)
	{
		// Calculate surface coordinates
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < width && y < height) {
			float2 data{};
			// Read from input surface
			uint16_t x16bits;
			surf2Dread(&x16bits, src, x * pixelSize + xOffset, y);
			data.x = x16bits * 1.0 / 65536.0f;
			// Write to output mat data
			float2* pixel = reinterpret_cast<float2*>(dst + (height - y - 1) * dstStep + x * 2 * sizeof(float));
			*pixel = data;
		}
	}

	// Assume src is in 8U and dst is 32FC2
	__global__ void
	copy8USurfaceToComplexMat(cudaSurfaceObject_t src,
			uchar* dst, size_t dstStep,
			size_t width, size_t height, size_t xOffset, size_t pixelSize)
	{
		// Calculate surface coordinates
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < width && y < height) {
			float2 data{};
			// Read from input surface
			uint8_t x8bit;
			surf2Dread(&x8bit, src, x * pixelSize + xOffset, y);
			data.x = x8bit * 1.0 / 256.0f;
			// Write to output mat data
			float2* pixel = reinterpret_cast<float2*>(dst + (height - y - 1) * dstStep + x * 2 * sizeof(float));
			*pixel = data;
		}
	}

	// Assumes input is 32FC2 and output 32FC1
	__global__ void
	copyComplexMatToSurface(uchar* src,
		cudaSurfaceObject_t dst, size_t srcStep,
		int width, int height)
	{
		// Calculate surface coordinates
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < width && y < height) {
			// Read from mat data
			float* data = reinterpret_cast<float*>(src + (height - y - 1) * srcStep + x * 2 * sizeof(float));
			// Write to surface
			surf2Dwrite(*data, dst, x * sizeof(float), y);
		}
	}

	__global__ void
	copyMatToSurface(uchar* src,
		cudaSurfaceObject_t dst, size_t srcStep,
		int width, int height)
	{
		// Calculate surface coordinates
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < width && y < height) {
			// Read from mat data
			uchar data = *(src + (height - y - 1) * srcStep + x);
			// Write to surface
			surf2Dwrite(data, dst, x, y);
		}
	}
}

// input is float channels and output is CV_32FC2
void 
GpuUtils::arrayToComplexMatGPU(int width, int height, cudaArray* input, cv::cuda::GpuMat& output, int numChannels, int channel, ChannelFormat cf)
{
	// Create the output surface object
	cudaSurfaceObject_t inputS{};
	createSurfaceObj(&inputS, input);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	switch (cf)
	{
	case GpuUtils::ChannelFormat::U8:
		copy8USurfaceToComplexMat<<<gridSize, blockSize>>>(inputS, output.data, output.step, width, height, channel * 1, numChannels * 1);
		break;
	case GpuUtils::ChannelFormat::U16:
		copy16USurfaceToComplexMat<<<gridSize, blockSize >> >(inputS, output.data, output.step, width, height, channel * 2, numChannels * 2);
		break;
	case GpuUtils::ChannelFormat::F16:
		copy16FSurfaceToComplexMat<<<gridSize, blockSize >> >(inputS, output.data, output.step, width, height, channel * 2, numChannels * 2);
		break;
	case GpuUtils::ChannelFormat::F32:
		copy32FSurfaceToComplexMat<<<gridSize, blockSize >> >(inputS, output.data, output.step, width, height, channel * 4, numChannels * 4);
		break;
	default:
		break;
	}

	cudaDestroySurfaceObject(inputS);
}

void 
GpuUtils::complexMatGPUToArray(int width, int height, const cv::cuda::GpuMat& input, cudaArray* output)
{
	// Create the output surface object
	cudaSurfaceObject_t outputS{};
	createSurfaceObj(&outputS, output);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	copyComplexMatToSurface<<<gridSize, blockSize>>>(input.data, outputS, input.step, width, height);

	cudaDestroySurfaceObject(outputS);
}

void 
GpuUtils::matGPUToArray(int width, int height, const cv::cuda::GpuMat& input, cudaArray* output, int pixelSize)
{
	width *= pixelSize;

	// Create the output surface object
	cudaSurfaceObject_t outputS{};
	createSurfaceObj(&outputS, output);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	uchar* inData = input.data;

	copyMatToSurface << <gridSize, blockSize >> >(inData, outputS, input.step, width, height);

	cudaDestroySurfaceObject(outputS);
}

void 
GpuUtils::arrayToMatGPU(int width, int height, cudaArray* input, cv::cuda::GpuMat& output, int pixelSize)
{
	width *= pixelSize;

	// Create the output surface object
	cudaSurfaceObject_t inputS{};
	createSurfaceObj(&inputS, input);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	uchar* outData = output.data;

	copySurfaceToMat << <gridSize, blockSize >> >(inputS, outData, output.step, width, height);

	cudaDestroySurfaceObject(inputS);
}