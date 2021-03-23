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

#include "DistanceTransformTOP.h"
#include "Parameters.h"

#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
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
}

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

DLLEXPORT
void
FillTOPPluginInfo(TOP_PluginInfo* info)
{
	// This must always be set to this constant
	info->apiVersion = TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TOP_ExecuteMode::CPUMemWriteOnly;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Distancetransfrom");
	// English readable name
	customInfo.opLabel->setString("Distance Transform");
	// Information of the author of the node
	customInfo.authorName->setString("Gabriel Robels");
	customInfo.authorEmail->setString("support@derivative.ca");

	// This TOP takes one input
	customInfo.minInputs = 1;
	customInfo.maxInputs = 1;
}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll
	return new DistanceTransformTOP(info);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (DistanceTransformTOP*)instance;
}

};


DistanceTransformTOP::DistanceTransformTOP(const OP_NodeInfo*) :
	myFrame{ new cv::Mat() }
{
}

DistanceTransformTOP::~DistanceTransformTOP()
{
	delete myFrame;
}

void
DistanceTransformTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
    ginfo->cookEveryFrame = false;
	ginfo->cookEveryFrameIfAsked = false;
	ginfo->memPixelType = OP_CPUMemPixelType::R32Float;
}

bool
DistanceTransformTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs* inputs, void*)
{
	// In this function we could assign variable values to 'format' to specify
	// the pixel format/resolution etc that we want to output to.
	// If we did that, we'd want to return true to tell the TOP to use the settings we've
	// specified.
	format->bitsPerChannel = 32;
	format->floatPrecision = true;
	format->redChannel = true;
	format->greenChannel = false;
	format->blueChannel = false;
	format->alphaChannel = false;

	// Query size of top, otherwise if 'Common/Output Resolution' is different than 'Use Input'
	// It scales the size twice
	const OP_TOPInput* top = inputs->getInputTOP(0);
	if (top)
	{
		format->width = top->width;
		format->height = top->height;
	}

	return true;
}

void
DistanceTransformTOP::execute(TOP_OutputFormatSpecs* output, const OP_Inputs* inputs, TOP_Context*, void*)
{
	using namespace cv;
	
	inputTopToMat(inputs);
	if (myFrame->empty())
		return;

	int distanceType = getType(myParms.evalDistancetype(inputs));
	int maskSize = getMask(myParms.evalMasksize(inputs));

	distanceTransform(*myFrame, *myFrame, distanceType, maskSize);
	
	if (myParms.evalNormalize(inputs))
		normalize(*myFrame, *myFrame, 0, 1.0, NORM_MINMAX);

	cvMatToOutput(output);
}

void
DistanceTransformTOP::setupParameters(OP_ParameterManager* in, void*)
{
	myParms.setup(in);
}

void
DistanceTransformTOP::cvMatToOutput(TOP_OutputFormatSpecs* out) const
{
	int	height = out->height;
	int	width = out->width;

	out->newCPUPixelDataLocation = 0;

	cv::resize(*myFrame, *myFrame, cv::Size(width, height));
	cv::flip(*myFrame, *myFrame, 0);

	memcpy(out->cpuPixelData[0], myFrame->data, height * width * sizeof(float));
}

void 
DistanceTransformTOP::inputTopToMat(const OP_Inputs* in)
{
	const OP_TOPInput*	top = in->getInputTOP(0);
	if (!top) {
                *myFrame = cv::Mat();
		return;
        }

	OP_TOPInputDownloadOptions	opts = {};
	opts.verticalFlip = true;
	opts.downloadType = static_cast<OP_TOPInputDownloadType>(myParms.evalDownloadtype(in));
	opts.cpuMemPixelType = OP_CPUMemPixelType::RGBA8Fixed;

	uint8_t*	pixel = (uint8_t*)in->getTOPDataInCPUMemory(top, &opts);
	if (!pixel) {
                *myFrame = cv::Mat();
		return;
        }

	int	height = top->height;
	int	width = top->width;

	*myFrame = cv::Mat(height, width, CV_8UC1);
	uint8_t*	data = (uint8_t*)myFrame->data;
        for (int i = 0; i < height; i += 1) {
                for (int j = 0; j < width; j += 1) {
                        int pixelN = i*width + j;
                        int index = 4*pixelN + static_cast<int>(myParms.evalChannel(in));
                        data[pixelN] = pixel[index];
                }
        }
}
