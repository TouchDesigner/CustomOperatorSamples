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

#include "SpectrumTOP.h"
#include "Parameters.h"
#include "GpuUtils.cuh"

#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/video/tracking.hpp>

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
	customInfo.opType->setString("Spectrum");
	// English readable name
	customInfo.opLabel->setString("Spectrum TOP");
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
	return new SpectrumTOP(info);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (SpectrumTOP*)instance;
}

};


SpectrumTOP::SpectrumTOP(const OP_NodeInfo*) :
	myFrame{ new cv::cuda::GpuMat() }, myResult{ new cv::cuda::GpuMat() }
{
}

SpectrumTOP::~SpectrumTOP()
{
	delete myFrame;
	delete myResult;
}

void
SpectrumTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	ginfo->cookEveryFrameIfAsked = false;
}

bool
SpectrumTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs* inputs, void*)
{
	// In this function we could assign variable values to 'format' to specify
	// the pixel format/resolution etc that we want to output to.
	// If we did that, we'd want to return true to tell the TOP to use the settings we've
	// specified.
	// In this example we want a single channel
	format->bitsPerChannel = 32;
	format->floatPrecision = true;
	format->redChannel = true;
	format->greenChannel = myParms.evalMode(inputs) == ModeMenuItems::dft;
	format->blueChannel = false;
	format->alphaChannel = false;

	// Query size of top, otherwise if 'Common/Output Resolution' is different than 'Use Input'
	// It scales the size twice
	const OP_TOPInput* top = inputs->getInputTOP(0);
	if (top)
	{
		format->width = mySize.width = top->width;
		format->height = mySize.height = top->height;
	}

	return true;
}

void
SpectrumTOP::execute(TOP_OutputFormatSpecs* output, const OP_Inputs* inputs, TOP_Context*, void*)
{
	using namespace cv::cuda;

	const OP_TOPInput* top = inputs->getInputTOP(0);
	if (!top || !checkInputTop(top))
		return;
	
	inputTopToMat(top);
	if (myFrame->empty())
		return;

	if (myParms.evalMode(inputs) == ModeMenuItems::dft)
	{
		dft(*myFrame, *myResult, mySize, 0);

		if (!myParms.evalTransrows(inputs))
		{
			swapQuadrants(*myResult);
		}
		else
		{
			swapSides(*myResult);
		}

		if (myParms.evalCoord(inputs) == CoordMenuItems::polar)
		{
			GpuMat channels[2];
			split(*myResult, channels);

			cartToPolar(channels[0], channels[1], channels[0], channels[1]);

			add(channels[0], 1, channels[0]);
			log(channels[0], channels[0]);

			merge(channels, 2, *myResult);
		}
	}
	else
	{
		if (myParms.evalCoord(inputs) == CoordMenuItems::polar)
		{
			GpuMat channels[2];
			split(*myFrame, channels);

			cv::cuda::exp(channels[0], channels[0]);
			add(channels[0], -1, channels[0]);
			polarToCart(channels[0], channels[1], channels[0], channels[1]);

			merge(channels, 2, *myFrame);
		}

		if (!myParms.evalTransrows(inputs))
		{
			swapQuadrants(*myFrame);
		}
		else
		{
			swapSides(*myFrame);
		}

		dft(*myFrame, *myResult, mySize, 0);
	}

	cvMatToOutput(*myResult, output);
}

void
SpectrumTOP::setupParameters(OP_ParameterManager* manager, void*)
{
	myParms.setup(manager);
}

void 
SpectrumTOP::getErrorString(OP_String* error, void*)
{
	error->setString(myError.c_str());
	myError.clear();
}

void 
SpectrumTOP::cvMatToOutput(const cv::cuda::GpuMat& M, const OP_Inputs* input, TOP_OutputFormatSpecs* out) const
{
	if (myParms.evalMode(input) == ModeMenuItems::dft)
	{
		GpuUtils::matGPUToArray(out->width, out->height, M, out->cudaOutput[0], 2 * sizeof(float));
	}
	else
	{
		GpuUtils::complexMatGPUToArray(out->width, out->height, M, out->cudaOutput[0]);
	}
}

void 
SpectrumTOP::inputTopToMat(const OP_TOPInput* top)
{
	*myFrame = cv::cuda::GpuMat(top->height, top->width, CV_32FC2);
	if (myParms.evalMode(top) == ModeMenuItems::dft)
	{
		GpuUtils::arrayToComplexMatGPU(top->width, top->height, top->cudaInput, *myFrame, myNumChan, static_cast<int>(myParms.evalChan(top)), myChanFormat);
	}
	else
	{
		GpuUtils::arrayToMatGPU(top->width, top->height, top->cudaInput, *myFrame, 2 * sizeof(float));
	}
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
SpectrumTOP::checkInputTop(const OP_TOPInput* topInput)
{
	ModeMenuItems myMode = myParms.evalMode(topInput);
	if (myMode == ModeMenuItems::idft && topInput->pixelFormat != RG32F)
	{
		myError = "Inverse transform requires a 32-bit float RG texture.";
		return false;
	}

	ChanMenuItems myChan = myParms.evalChan(topInput);
	switch (topInput->pixelFormat)
	{
		case A8:
		case A16:
		case A16F:
		case A32F:
			// Only A channel is valid, change to use channel as index
			if (myChan == ChanMenuItems::a)
				myChan == ChanMenuItems::r;
			else
				myChan = ChanMenuItems::r; // ::Invalid what is invalid here ?
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
			// Only RA channels are valid, change to use channel as index
			if (myChan == ChanMenuItems::a)
				myChan = ChanMenuItems::r; // ::Second what is Second here ?
			else if (myParms.evalChan(topInput) != ChanMenuItems::r)
				myChan = ChanMenuItems::r;  // ::Invalid what is Invalid here ?
		case RG8:
		case RG16:
		case RG16F:
		case RG32F:
			myNumChan = 2;
			break;
		// RGB has alpha on its channels
		case RGB16F:
		case RGB32F:
			if (myChan == ChanMenuItems::a)
				myChan = ChanMenuItems::r; // ::Invalid what is Invalid here ?
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
			myChanFormat = GpuUtils::ChannelFormat::U8;
			break;
		case A16F:
		case R16F:
		case RA16F:
		case RG16F:
		case RGB16F:
		case RGBA16F:
			myChanFormat = GpuUtils::ChannelFormat::F16;
			break;
		case A16:
		case R16:
		case RA16:
		case RG16:
		case RGBA16:
			myChanFormat = GpuUtils::ChannelFormat::U16;
			break;
		case A32F:
		case R32F:
		case RA32F:
		case RG32F:
		case RGB32F:
		case RGBA32F:
			myChanFormat = GpuUtils::ChannelFormat::F32;
			break;
	}

	if (myMode == ModeMenuItems::dft && static_cast<int>(myChan) >= myNumChan)
	{
		myError = "Channel not available.";
		return false;
	}

	if (topInput->cudaInput == nullptr)
	{
		myError = "CUDA memory for input TOP was not mapped correctly.";
		return false;
	}
	return true;
}

void SpectrumTOP::swapQuadrants(cv::cuda::GpuMat& mat)
{
	using namespace cv::cuda;

	int cx = mySize.width / 2;
	int cy = mySize.height / 2;

	GpuMat q0(mat, cv::Rect(0, 0, cx, cy));
	GpuMat q1(mat, cv::Rect(cx, 0, cx, cy));
	GpuMat q2(mat, cv::Rect(0, cy, cx, cy));
	GpuMat q3(mat, cv::Rect(cx, cy, cx, cy));

	GpuMat tmp;
	// swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	// swap quadrant (Top-Right with Bottom-Left)
	q1.copyTo(tmp);                    
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void SpectrumTOP::swapSides(cv::cuda::GpuMat& mat)
{
	using namespace cv::cuda;

	int cx = mySize.width / 2;

	GpuMat right(mat, cv::Rect(0, 0, cx, mySize.height));
	GpuMat left(mat, cv::Rect(cx, 0, cx, mySize.height));

	GpuMat tmp;
	right.copyTo(tmp);
	left.copyTo(right);
	tmp.copyTo(left);
}
