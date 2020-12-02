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

#include <cstdint>
#include <array>

// Simple copy surface
__global__ void 
copySurface(cudaSurfaceObject_t inputSurfObj,
	cudaSurfaceObject_t outputSurfObj,
	int width, int height)
{
	// Calculate surface coordinates
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height) {
		uchar4 data;
		// Read from input surface
		surf2Dread(&data, inputSurfObj, x * 4, y);
		// Write to output surface
		surf2Dwrite(data, outputSurfObj, x * 4, y);
	}
}

__global__ void
makeColorSurface(cudaSurfaceObject_t outputSurfObj,
	int width, int height, uchar4 color)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		// Write to output surface
		surf2Dwrite(color, outputSurfObj, x * 4, y);
	}
}

void createSurfaceObj(cudaSurfaceObject_t* sObj, cudaArray* arr)
{
	cudaResourceDesc resDesc = {};
	resDesc.res.array.array = arr;
	resDesc.resType = cudaResourceTypeArray;
	cudaCreateSurfaceObject(sObj, &resDesc);
}

void doCUDAOperation(int width, int height, cudaArray *input, cudaArray *output, std::array<uint8_t, 4> color)
{
	// Create the output surface object
	cudaSurfaceObject_t outputS{};
	createSurfaceObj(&outputS, output);

	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y);

	if (input)
	{
		cudaSurfaceObject_t inputS{};
		createSurfaceObj(&inputS, input);
		copySurface<<<gridSize, blockSize>>>(inputS, outputS, width, height);
		cudaDestroySurfaceObject(inputS);
	}
	else
	{
		uchar4 pixel = make_uchar4(color.at(0), color.at(1), color.at(2), color.at(3));
		makeColorSurface<<<gridSize, blockSize>>> (outputS, width, height, pixel);
	}

	cudaDestroySurfaceObject(outputS);
}
