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
#include <opencv2/imgproc.hpp>
#include <string>

#pragma region Menus
enum class DistancetypeMenuItems
{
	L1,
	L2,
	C
};

enum class MasksizeMenuItems
{
	Three,
	Five,
	Precise
};

enum class DownloadtypeMenuItems
{
	Delayed,
	Instant
};

enum class ChannelMenuItems
{
	R,
	G,
	B,
	A
};

#pragma endregion

/*
This example implements a TOP to calculate the distance transform using openCV.

It takes the following parameters:
	- Distance Type:        One of [L1, L2, C], which determines how to calculate the distance.
	- Mask Size:        One of [3x3, 5x5, Precise], which determines the size of the transform mask.
	- Normalize:	If On, normalize the output image.
For more information visit: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042

This TOP takes one input which must be 8 bit single channel.
*/

// To get more help about these functions, look at TOP_CPlusPlusBase.h
class DistanceTransformTOP : public TD::TOP_CPlusPlusBase
{
public:
    DistanceTransformTOP(const TD::OP_NodeInfo *info, TD::TOP_Context *context);
    virtual ~DistanceTransformTOP();

    virtual void		getGeneralInfo(TD::TOP_GeneralInfo*, const TD::OP_Inputs*, void* reserved) override;

    virtual void		execute(TD::TOP_Output*, const TD::OP_Inputs*, void* reserved) override;

	virtual void		setupParameters(TD::OP_ParameterManager*, void* reserved) override;

private:
	void                inputTopToMat(const TD::OP_Inputs*);

	void 				cvMatToOutput(TD::TOP_Output*, TD::TOP_UploadInfo) const;

	int getType(DistancetypeMenuItems dt)
	{
		switch (dt)
		{
		default:
		case DistancetypeMenuItems::L1:
			return cv::DIST_L1;
		case DistancetypeMenuItems::L2:
			return cv::DIST_L2;
		case DistancetypeMenuItems::C:
			return cv::DIST_C;
		}
	}

	int getMask(MasksizeMenuItems ms)
	{
		switch (ms)
		{
		default:
		case MasksizeMenuItems::Three:
			return cv::DIST_MASK_3;
		case MasksizeMenuItems::Five:
			return cv::DIST_MASK_5;
		case MasksizeMenuItems::Precise:
			return cv::DIST_MASK_PRECISE;
		}
	}

	cv::Mat*		myFrame;

	int					myExecuteCount;
	TD::TOP_Context* myContext;
	TD::OP_SmartRef<TD::OP_TOPDownloadResult> myPrevDownRes;
};

#endif
