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

#include "OpticalFlowCPUTOP.h"
#include "Parameters.h"

#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

DLLEXPORT
void
FillTOPPluginInfo(TOP_PluginInfo *info)
{
	// This must always be set to this constant
	info->apiVersion = TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TOP_ExecuteMode::CPUMemWriteOnly;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Opticalflowcpu");
	// English readable name
	customInfo.opLabel->setString("Optical Flow CPU");
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
	return new OpticalFlowCPUTOP(info);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (OpticalFlowCPUTOP*)instance;
}

};


OpticalFlowCPUTOP::OpticalFlowCPUTOP(const OP_NodeInfo*) :
	myFrame{ new cv::Mat() }, myPrev{ new cv::Mat() }, myFlow{ new cv::Mat() }
{
}

OpticalFlowCPUTOP::~OpticalFlowCPUTOP()
{
	delete myFrame;
	delete myPrev;
	delete myFlow;
}

void
OpticalFlowCPUTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	ginfo->cookEveryFrameIfAsked = false;
	ginfo->memPixelType = OP_CPUMemPixelType::RG32Float;
	ginfo->inputSizeIndex = 0;
}

bool
OpticalFlowCPUTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs* inputs, void*)
{
	// In this function we could assign variable values to 'format' to specify
	// the pixel format/resolution etc that we want to output to.
	// If we did that, we'd want to return true to tell the TOP to use the settings we've
	// specified.
	// In this example we want a dual channel for output uv vectors
	format->bitsPerChannel = 32;
	format->floatPrecision = true;
	format->redChannel = true;
	format->greenChannel = true;
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
OpticalFlowCPUTOP::execute(TOP_OutputFormatSpecs* output, const OP_Inputs* inputs, TOP_Context*, void*)
{
	using namespace cv;

	inputToMat(inputs);
	if (myFrame->empty())
		return;

	Size outSize = Size(output->width, output->height);
	resize(*myFrame, *myFrame, outSize);
	
	if (myPrev->empty() || myPrev->size() != outSize)
	{
		*myPrev = std::move(*myFrame);
		return;
	}

	int myFlags = myParms.evalUsegaussianfilter(inputs) ? cv::OPTFLOW_FARNEBACK_GAUSSIAN : 0;
	myFlags |= myParms.evalUsepreviousflow(inputs) ? cv::OPTFLOW_USE_INITIAL_FLOW : 0;

	if (myFlow->empty() || myFlow->size() != outSize)
	{
		*myFlow = Mat(outSize, CV_32FC2);
		myFlags &= ~OPTFLOW_USE_INITIAL_FLOW;
	}

	calcOpticalFlowFarneback(
		*myPrev, *myFrame, *myFlow, myParms.evalPyramidscale(inputs), myParms.evalNumlevels(inputs), myParms.evalWindowsize(inputs),
		myParms.evalIterations(inputs), myParms.evalPolyn(inputs), myParms.evalPolysigma(inputs), myFlags
	);

	*myPrev = std::move(*myFrame);

	cvMatToOutput(*myFlow, output);
}

void
OpticalFlowCPUTOP::setupParameters(OP_ParameterManager* manager, void*)
{
	myParms.setup(manager);
}

void 
OpticalFlowCPUTOP::cvMatToOutput(const cv::Mat& M, TOP_OutputFormatSpecs* out) const
{
	size_t	height = out->height;
	size_t	width = out->width;

	out->newCPUPixelDataLocation = 0;
	float*	pixel = static_cast<float*>(out->cpuPixelData[0]);
	
	cv::flip(M, M, 0);
	float*	data = static_cast<float*>(static_cast<void*>(M.data));

	memcpy(pixel, data, 2 * height * width * sizeof(float));
}

void 
OpticalFlowCPUTOP::inputToMat(const OP_Inputs* in) const
{
	const OP_TOPInput*	top = in->getInputTOP(0);
	if (!top)
        {
                *myFrame = cv::Mat();
		return;
        }

	OP_TOPInputDownloadOptions	opts = {};
	opts.verticalFlip = true;
	opts.downloadType = myParms.evalDownloadtype(in);
	opts.cpuMemPixelType = OP_CPUMemPixelType::RGBA8Fixed;

	uint8_t*	pixel = (uint8_t*)in->getTOPDataInCPUMemory(top, &opts);
	if (!pixel)
        {
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
