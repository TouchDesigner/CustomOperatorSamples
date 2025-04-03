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

#include "CPlusPlus_Common.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <stdio.h>

__global__ void
copyRGBATToRGBPlanar(int width, int height, cudaSurfaceObject_t input, int64_t planeStride, float* output)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	uchar4 color;
	surf2Dread(&color, input, x * 4, y, cudaBoundaryModeZero);
	int64_t pixelIndex = y * width + x;
	output[pixelIndex] = (float)color.x;
	output[pixelIndex + planeStride] = (float)color.y;
	output[pixelIndex + planeStride * 2] = (float)color.z;
}

__device__ unsigned char
clampFloat(float v)
{
	if (v < 0.0)
		return 0;
	if (v > 255.0)
		return 255;
	else
		return (unsigned char)v;
}


__global__ void
copyRGBPlanarToRGBA(int width, int height, float* input, int64_t planeStride, cudaSurfaceObject_t output)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	uchar4 color;
	int64_t pixelIndex = y * width + x;
	color.x = clampFloat(input[pixelIndex]);
	color.y = clampFloat(input[pixelIndex + planeStride]);
	color.z = clampFloat(input[pixelIndex + planeStride * 2]);
	color.w = 255;
	surf2Dwrite(color, output, x * 4, y, cudaBoundaryModeZero);
}

int
divUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

cudaError_t
doCopyRGBATToRGBPlanar(int width, int height, int depth, TD::OP_TexDim dim, cudaSurfaceObject_t input, float* output, cudaStream_t stream)
{
	cudaError_t cudaStatus;

	dim3 blockSize(16, 16, 1);
	dim3 gridSize(divUp(width, blockSize.x), divUp(height, blockSize.y), depth);

	copyRGBATToRGBPlanar << <gridSize, blockSize, 0, stream >> > (width, height, input, width * height, output);

#ifdef _DEBUG
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
	}
#else
	cudaStatus = cudaSuccess;
#endif

	return cudaStatus;
}

cudaError_t
doCopyRGBPlanarToRGBA(int width, int height, int depth, TD::OP_TexDim dim, float* input, cudaSurfaceObject_t output, cudaStream_t stream)
{
	cudaError_t cudaStatus;

	dim3 blockSize(16, 16, 1);
	dim3 gridSize(divUp(width, blockSize.x), divUp(height, blockSize.y), depth);

	copyRGBPlanarToRGBA << <gridSize, blockSize, 0, stream >> > (width, height, input, width * height, output);

#ifdef _DEBUG
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
	}
#else
	cudaStatus = cudaSuccess;
#endif

	return cudaStatus;
}
