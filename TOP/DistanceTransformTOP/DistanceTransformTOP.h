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

#ifndef __DistanceTransformTOP__
#define __DistanceTransformTOP__

#include "TOP_CPlusPlusBase.h"

#include <opencv2\core.hpp>
#include <string>

struct Parameters;

/*
This example implements a TOP to calculate the distance transform using openCV's functionallity.

It takes the following parameters:
	- Distance Type:        One of [L1, L2, C], which determines how to calculate the distance.
	- Mask Size:        One of [3x3, 5x5, Precise], which determines the size of the transform mask.
	- Normalize:	If On, normalize the output image.
For more information visit: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042

This TOP takes one input which must be 8 bit single channel.
*/

// To get more help about these functions, look at TOP_CPlusPlusBase.h
class DistanceTransformTOP : public TOP_CPlusPlusBase
{
public:
    DistanceTransformTOP(const OP_NodeInfo *info);
    virtual ~DistanceTransformTOP();

    virtual void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved) override;

    virtual bool		getOutputFormat(TOP_OutputFormat*, const OP_Inputs*, void* reserved) override;

    virtual void		execute(TOP_OutputFormatSpecs*, const OP_Inputs*, TOP_Context*, void* reserved) override;

	virtual void		setupParameters(OP_ParameterManager*, void* reserved) override;

private:
	void                inputTopToMat(const OP_Inputs*);

	void 				cvMatToOutput(TOP_OutputFormatSpecs*) const;

	Parameters*		myParms;

	cv::Mat*		myFrame;
};

#endif
