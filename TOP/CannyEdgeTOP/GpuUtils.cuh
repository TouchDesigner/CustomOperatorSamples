#ifndef __GPUUTILS__
#define __GPUUTILS__

struct cudaArray;
namespace cv
{
	namespace cuda
	{
		class GpuMat;
	}
}

namespace GpuUtils
{
	enum class ChannelFormat
	{
		U8,
		U16,
		F16,
		F32
	};

	void	matGPUToArray(int width, int height, const cv::cuda::GpuMat& input, cudaArray* output, int pixelSize);

	void	arrayToMatGPU(int width, int height, cudaArray* input, cv::cuda::GpuMat& output, int pixelSize);
}
#endif
