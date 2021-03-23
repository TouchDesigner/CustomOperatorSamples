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

#ifndef __SpectrumTOP__
#define __SpectrumTOP__

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
This example implements a TOP to calculate the fourier transform using openCV's cuda functionallity.

It takes the following parameters:
	- Transform:	One of [Image To DFT, DFT To Image], which determines if we calculate the forward or 
		inverse fourier transform.
	- Coordinate System:	One of [Polar, Cartesian]. If the transform is Image To DFT the output will be
		in the selected coordinate system. If the transform is DFT To Image the input must be in the 
		coordinate system selected.
	- Channel:	Active when Transform is Image To DFT. Selects which channel will be used to calculate the transform.
	- Per Rows:	If On, it calculates the fourier transform of each row independently.

This TOP takes one input. If the inverse is to be calculated the input must have
exactly 2 32-bit float channels.
*/

// To get more help about these functions, look at TOP_CPlusPlusBase.h
class SpectrumTOP : public TOP_CPlusPlusBase
{
public:
    SpectrumTOP(const OP_NodeInfo *info);
    virtual ~SpectrumTOP();

    virtual void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved) override;

    virtual bool		getOutputFormat(TOP_OutputFormat*, const OP_Inputs*, void* reserved) override;

    virtual void		execute(TOP_OutputFormatSpecs*, const OP_Inputs*, TOP_Context*, void* reserved) override;

	virtual void		setupParameters(OP_ParameterManager*, void* reserved) override;

	virtual void		getErrorString(OP_String*, void* reserved) override;

private:
    void                inputTopToMat(const OP_TOPInput*);

	void 				cvMatToOutput(const cv::cuda::GpuMat&, const OP_Inputs* input, TOP_OutputFormatSpecs*) const;

	bool				checkInputTop(const OP_TOPInput*);

	void				swapQuadrants(cv::cuda::GpuMat&);

	void				swapSides(cv::cuda::GpuMat&);

	cv::cuda::GpuMat*	myFrame;
	cv::cuda::GpuMat*	myResult;

	Parameters			myParms;
	std::string			myError;

	int						myNumChan;
	GpuUtils::ChannelFormat	myChanFormat;

	cv::Size			mySize;
};

#endif
