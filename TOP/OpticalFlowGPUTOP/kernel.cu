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

#include "cuda_runtime.h"

#include <opencv2/core/cuda.hpp>

// Assume data is in rgba8
__global__ void 
copySurfaceToMat(cudaSurfaceObject_t src,
	uchar* dst, size_t dstStep,
	int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		uchar4 data;
		// Read from input surface
		surf2Dread(&data, src, x * 4, y);
		// Write to output mat data
		uchar* pixel = dst + (height - y - 1) * dstStep + x * 4;
		pixel[0] = data.z;
		pixel[1] = data.y;
		pixel[2] = data.x;
		pixel[3] = data.w;
	}
}

// Assumes data is in float with 32 bits 2 channels
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
		uchar* pixel = src + (height - y - 1) * srcStep + x * 2 * sizeof(float);
		float2 data = *reinterpret_cast<float2*>(pixel);
		// Write to surface
		surf2Dwrite(data, dst, x * sizeof(float2), y);
	}
}

void createSurfaceObj(cudaSurfaceObject_t* sObj, cudaArray* arr)
{
	cudaResourceDesc resDesc = {};
	resDesc.res.array.array = arr;
	resDesc.resType = cudaResourceTypeArray;
	cudaCreateSurfaceObject(sObj, &resDesc);
}

void arrayToMatGPU8UC4(int width, int height, cudaArray* input, cv::cuda::GpuMat& output)
{
	// Create the output surface object
	cudaSurfaceObject_t inputS{};
	createSurfaceObj(&inputS, input);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	uchar* outData = output.data;
	// Assume data is in rgba8
	copySurfaceToMat<<<gridSize, blockSize>>>(inputS, outData, output.step, width, height);

	cudaDestroySurfaceObject(inputS);
}

void matGPUToArray32FC2(int width, int height, const cv::cuda::GpuMat& input, cudaArray* output)
{
	// Create the output surface object
	cudaSurfaceObject_t outputS{};
	createSurfaceObj(&outputS, output);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	uchar* inData = input.data;
	// Assumes data is in float with 32 bits 2 channels
	copyMatToSurface<<<gridSize, blockSize>>>(inData, outputS, input.step, width, height);

	cudaDestroySurfaceObject(outputS);
}
