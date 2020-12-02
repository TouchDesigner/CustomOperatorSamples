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
#ifndef __OpticalFlowGPUTOP__
#define __OpticalFlowGPUTOP__

#include "TOP_CPlusPlusBase.h"

#include <string>

namespace cv
{
	namespace cuda
	{
		class GpuMat;
	}
}

/*
This example implements a TOP to expose cv::calcOpticalFlowFarneback. For
more information on the parameters check 
https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af.

It takes the following parameters:
	- Num Levels:	Number of pyramid layers including the intial image.
	- Pyramid Scale:	Image scale to build pyramid layers.
	- Fast Pyramids:	If on, use fast pyramids.
	- Window Size:	Averaging window size.
	- Iterations:	Number of iteration at each pyramid level.
	- Poly N:	Size of the pixel neighborhood used to find polynomial expansion in each pixel.
	- Poly Sigma:	Standard deviation of the Gaussian thta is used to smooth derivatives used as
		basis for the polynomial expansion.
	- Use Gaussian Filter:	Uses the Gaussian Window Size x Window Size filter instead of a box filter.

This TOP takes one input where the optical flow of sequencial frames is calculated.
*/

// To get more help about these functions, look at TOP_CPlusPlusBase.h
class OpticalFlowGPUTOP : public TOP_CPlusPlusBase
{
public:
    OpticalFlowGPUTOP(const OP_NodeInfo *info);
    virtual ~OpticalFlowGPUTOP();

    virtual void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved) override;

    virtual bool		getOutputFormat(TOP_OutputFormat*, const OP_Inputs*, void* reserved) override;

    virtual void		execute(TOP_OutputFormatSpecs*, const OP_Inputs*, TOP_Context*, void* reserved) override;

	virtual void		setupParameters(OP_ParameterManager*, void* reserved) override;

	virtual void		getErrorString(OP_String*, void* reserved) override;

private:
    void                handleParameters(const OP_Inputs*);

    void                inputTopToMat(const OP_TOPInput*) const;

	void 				cvMatToOutput(const cv::cuda::GpuMat&, TOP_OutputFormatSpecs*) const;

	bool				checkInputTop(const OP_TOPInput*);

	cv::cuda::GpuMat*	myFrame;
	cv::cuda::GpuMat*	myPrev;
	cv::cuda::GpuMat*	myFlow;

	std::string			myError;

    // Parameters
	int		myNumLevels;
	double	myPyrScale;
	bool	myFastPyramids;
	int		myWinSize;
	int		myNumIter;
	int		myPolyN;
	double	myPolySigma;
	int		myFlags;
};

#endif
