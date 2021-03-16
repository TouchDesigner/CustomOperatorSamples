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

#include "CannyEdgeTOP.h"
#include "Parameters.h"
#include "GpuUtils.cuh"

#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include "cuda_runtime.h"

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
	info->executeMode = TOP_ExecuteMode::CUDA;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Cannyedge");
	// English readable name
	customInfo.opLabel->setString("Canny Edge");
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
	return new CannyEdgeTOP(info);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (CannyEdgeTOP*)instance;
}

};


CannyEdgeTOP::CannyEdgeTOP(const OP_NodeInfo*) :
	myFrame{ new cv::cuda::GpuMat() }
{
}

CannyEdgeTOP::~CannyEdgeTOP()
{
	delete myFrame;
}

void
CannyEdgeTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	ginfo->cookEveryFrameIfAsked = false;
}

bool
CannyEdgeTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs* inputs, void*)
{
	// In this function we could assign variable values to 'format' to specify
	// the pixel format/resolution etc that we want to output to.
	// If we did that, we'd want to return true to tell the TOP to use the settings we've
	// specified.
	// In this example we want a single channel
	format->bitsPerChannel = 8;
	format->floatPrecision = false;
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
CannyEdgeTOP::execute(TOP_OutputFormatSpecs* output, const OP_Inputs* inputs, TOP_Context*, void*)
{
	using namespace cv::cuda;

	const OP_TOPInput* top = inputs->getInputTOP(0);
	if (!top || !checkInputTop(top))
		return;
	
	inputTopToMat(top);
	if (myFrame->empty())
		return;

	int appertureSize = myParms.evalApperturesize(inputs);
	if (appertureSize % 2 == 0)
		++appertureSize;

	myFrame->convertTo(*myFrame, CV_8U);
	if (myNumChan > 1)
	{
		GpuMat chan[4];
		split(*myFrame, chan);
		*myFrame = chan[0];
	}
	auto cannyEdge = createCannyEdgeDetector(255 * myParms.evalLowthreshold(inputs), 255 * myParms.evalHighthreshold(inputs), myParms.evalApperturesize(inputs), myParms.evalL2gradient(inputs));
	cannyEdge->detect(*myFrame, *myFrame);

	cvMatToOutput(output);
}

void
CannyEdgeTOP::setupParameters(OP_ParameterManager* in, void*)
{
	myParms.setup(in);
}

void 
CannyEdgeTOP::getErrorString(OP_String* error, void*)
{
	error->setString(myError.c_str());
	myError.clear();
}

void 
CannyEdgeTOP::cvMatToOutput(TOP_OutputFormatSpecs* out) const
{
	GpuUtils::matGPUToArray(out->width, out->height, *myFrame, out->cudaOutput[0], 1);
}

void 
CannyEdgeTOP::inputTopToMat(const OP_TOPInput* top)
{
	*myFrame = cv::cuda::GpuMat(top->height, top->width, myMatType);
	GpuUtils::arrayToMatGPU(top->width, top->height, top->cudaInput, *myFrame, myPixelSize);
}

// Definitions for pixel formats
#define RGBA8 0x8058
#define RGBA16F 0x881A
#define RGBA32F 0x8814
#define RGBA16 0x805B
#define RGB16F 0xF000E
#define RGB32F 0xF000F
#define R8 0x8229
#define R16 0x822A
#define R16F 0x822D
#define R32F 0x822E
#define RG8 0x822B
#define RG16 0x822C
#define RG16F 0x822F
#define RG32F 0x8230
#define A8 0xF0001
#define A16 0xF0002
#define A16F 0xF0003
#define A32F 0xF0004
#define RA8 0xF0005
#define RA16 0xF0006
#define RA16F 0xF0007
#define RA32F 0xF0008

bool
CannyEdgeTOP::checkInputTop(const OP_TOPInput* topInput)
{
	switch (topInput->pixelFormat)
	{
	case A8:
	case A16:
	case A16F:
	case A32F:
	case R8:
	case R16:
	case R16F:
	case R32F:
		myNumChan = 1;
		break;
	case RA8:
	case RA16:
	case RA16F:
	case RA32F:
	case RG8:
	case RG16:
	case RG16F:
	case RG32F:
		myNumChan = 2;
		break;
		// RGB has alpha on its channels
	case RGB16F:
	case RGB32F:
	case RGBA8:
	case RGBA16:
	case RGBA16F:
	case RGBA32F:
		myNumChan = 4;
		break;
	default:
		myError = "Pixel format not supported.";
		myNumChan = -1;
		return false;
	}

	switch (topInput->pixelFormat)
	{
	case A8:
	case R8:
	case RA8:
	case RG8:
	case RGBA8:
		myMatType = CV_MAKETYPE(CV_8U, myNumChan);
		myPixelSize = 1 * myNumChan;
		break;
	case A16F:
	case R16F:
	case RA16F:
	case RG16F:
	case RGB16F:
	case RGBA16F:
		myMatType = CV_MAKETYPE(CV_16F, myNumChan);
		myPixelSize = 2 * myNumChan;
		break;
	case A16:
	case R16:
	case RA16:
	case RG16:
	case RGBA16:
		myMatType = CV_MAKETYPE(CV_16U, myNumChan);
		myPixelSize = 2 * myNumChan;
		break;
	case A32F:
	case R32F:
	case RA32F:
	case RG32F:
	case RGB32F:
	case RGBA32F:
		myMatType = CV_MAKETYPE(CV_32F, myNumChan);
		myPixelSize = 2 * myNumChan;
		break;
	}

	if (topInput->cudaInput == nullptr)
	{
		myError = "CUDA memory for input TOP was not mapped correctly.";
		return false;
	}
	return true;
}
