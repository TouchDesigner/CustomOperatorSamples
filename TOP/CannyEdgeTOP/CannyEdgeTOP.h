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

#ifndef __CannyEdgeTOP__
#define __CannyEdgeTOP__

#include "TOP_CPlusPlusBase.h"
#include "Parameters.h"

#include <opencv2\core.hpp>
#include <string>

namespace cv
{
	namespace cuda
	{
		class GpuMat;
	}
}

namespace GpuUtils
{
	enum class ChannelFormat;
}

/*
This example implements a TOP exposing the canny edge detector using OpenCV's CUDA functionallity.

It takes the following parameters:
		- Low Threshold:    First threshold for the hysteresis procedure.
		- High Threshold:   Second threshold for the hysteresis procedure.
		- Aperture size:    Aperture size for the Sobel operator.
		- L2 Gradient:  If On, a more accurate norm should be used to compute the image gradient.
For more information visit: https://docs.opencv.org/3.4/d0/d05/group__cudaimgproc.html#gabc17953de36faa404acb07dc587451fc

This TOP takes one input which must be 8 bit single channel.
*/

// To get more help about these functions, look at TOP_CPlusPlusBase.h
class CannyEdgeTOP : public TD::TOP_CPlusPlusBase
{
public:
    CannyEdgeTOP(const TD::OP_NodeInfo *info, TD::TOP_Context *context);
    virtual ~CannyEdgeTOP();

    virtual void		getGeneralInfo(TD::TOP_GeneralInfo*, const TD::OP_Inputs*, void* reserved) override;

    virtual void		execute(TD::TOP_Output*, const TD::OP_Inputs*, void* reserved) override;

	virtual void		setupParameters(TD::OP_ParameterManager*, void* reserved) override;

	virtual void		getErrorString(TD::OP_String*, void* reserved) override;

private:
	bool				checkInputTop(const TD::OP_TOPInput*);

	cv::cuda::GpuMat*	myFrame;

	std::string			myError;

	int					myNumChan;
	int					myMatType;
	int					myPixelSize;

	TD::TOP_Context*	myContext;
	cudaStream_t		myStream;

	Parameters			myParms;

	// In this example this value will be incremented each time the execute()
// function is called, then passes back to the TOP 
	int32_t				myExecuteCount;
};

#endif
