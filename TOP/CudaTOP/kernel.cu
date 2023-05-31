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
copyTextureRGBA82D(int width, int height, cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	uchar4 color;
	surf2Dread(&color, input, x * 4, y, cudaBoundaryModeZero);
	surf2Dwrite(color, output, x * 4, y, cudaBoundaryModeZero);
}

__global__ void
copyTextureRGBA83D(int width, int height, cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= width || y >= height)
		return;

	uchar4 color;
	surf3Dread(&color, input, x * 4, y, z, cudaBoundaryModeZero);
	surf3Dwrite(color, output, x * 4, y, z, cudaBoundaryModeZero);
}

__global__ void
copyTextureRGBA8Cube(int width, int height, cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= width || y >= height)
		return;

	uchar4 color;
	surfCubemapread(&color, input, x * 4, y, z, cudaBoundaryModeZero);
	surfCubemapwrite(color, output, x * 4, y, z, cudaBoundaryModeZero);
}

__global__ void
copyTextureRGBA82DArray(int width, int height, cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= width || y >= height)
		return;

	uchar4 color;
	surf2DLayeredread(&color, input, x * 4, y, z, cudaBoundaryModeZero);
	surf2DLayeredwrite(color, output, x * 4, y, z, cudaBoundaryModeZero);
}

__global__ void
fillOutput(int width, int height, uchar4 color, cudaSurfaceObject_t output)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	surf2Dwrite(color, output, x * 4, y, cudaBoundaryModeZero);
}

int
divUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

cudaError_t
doCUDAOperation(int width, int height, int depth, TD::OP_TexDim dim, float4 color, cudaSurfaceObject_t input, cudaSurfaceObject_t output)
{
	cudaError_t cudaStatus;

	dim3 blockSize(16, 16, 1);
	dim3 gridSize(divUp(width, blockSize.x), divUp(height, blockSize.y), depth);

	if (input)
	{
		switch (dim)
		{
			case TD::OP_TexDim::e2D:
				copyTextureRGBA82D << <gridSize, blockSize >> > (width, height, input, output);
				break;
			case TD::OP_TexDim::e3D:
				copyTextureRGBA83D << <gridSize, blockSize >> > (width, height, input, output);
				break;
			case TD::OP_TexDim::eCube:
				gridSize.z = 6;
				copyTextureRGBA8Cube << <gridSize, blockSize >> > (width, height, input, output);
				break;
			case TD::OP_TexDim::e2DArray:
				copyTextureRGBA82DArray << <gridSize, blockSize >> > (width, height, input, output);
				break;
		}
	}
	else
	{
		dim3 gridSize(divUp(width, blockSize.x), divUp(height, blockSize.y), 1);
		uchar4 c8;
		// Flip R and B since we are outputting to BGRA8
		c8.z = (uint8_t)std::min(std::max(color.x * 255.0f, 0.0f), 255.0f);
		c8.y = (uint8_t)std::min(std::max(color.y * 255.0f, 0.0f), 255.0f);
		c8.x = (uint8_t)std::min(std::max(color.z * 255.0f, 0.0f), 255.0f);
		c8.w = (uint8_t)std::min(std::max(color.w * 255.0f, 0.0f), 255.0f);
		fillOutput<<<gridSize, blockSize>>> (width, height, c8, output);
	}


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
