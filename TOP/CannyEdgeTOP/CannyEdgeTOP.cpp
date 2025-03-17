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
#include "GpuUtils.cuh"
#include "Parameters.h"

#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "cuda_runtime.h"

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

DLLEXPORT
void
FillTOPPluginInfo(TD::TOP_PluginInfo* info)
{
	// This must always be set to this constant
	info->apiVersion = TD::TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TD::TOP_ExecuteMode::CUDA;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	TD::OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Cannyedge");
	// English readable name
	customInfo.opLabel->setString("Canny Edge");
	// Information of the author of the node
	customInfo.authorName->setString("Author Name");
	customInfo.authorEmail->setString("email@email.ca");

	// This TOP takes one input
	customInfo.minInputs = 1;
	customInfo.maxInputs = 1;
}

DLLEXPORT
TD::TOP_CPlusPlusBase*
CreateTOPInstance(const TD::OP_NodeInfo* info, TD::TOP_Context* context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll
	return new CannyEdgeTOP(info, context);
}

DLLEXPORT
void
DestroyTOPInstance(TD::TOP_CPlusPlusBase* instance, TD::TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (CannyEdgeTOP*)instance;
}

};


CannyEdgeTOP::CannyEdgeTOP(const TD::OP_NodeInfo*, TD::TOP_Context *context) :
	myFrame{ new cv::cuda::GpuMat() },
	myExecuteCount(0),
	myError(""),
	myContext(context),
	myStream(0)
{
	cudaStreamCreate(&myStream);
}

CannyEdgeTOP::~CannyEdgeTOP()
{
	delete myFrame;
	if(myStream)
		cudaStreamDestroy(myStream);
}

void
CannyEdgeTOP::getGeneralInfo(TD::TOP_GeneralInfo* ginfo, const TD::OP_Inputs*, void*)
{
	ginfo->cookEveryFrameIfAsked = false;
}

void
CannyEdgeTOP::execute(TD::TOP_Output* output, const TD::OP_Inputs* inputs, void*)
{
	myError = "";
	myExecuteCount++;

	using namespace cv::cuda;

	const TD::OP_TOPInput* top = inputs->getInputTOP(0);
	if (!top || !checkInputTop(top))
		return;


	if (top->textureDesc.pixelFormat != TD::OP_PixelFormat::Mono8Fixed)
	{
		myError = "Input must be 8 bit single channel";
		return;
	}

	int appertureSize = myParms.evalApperturesize(inputs);
	double lothresh = myParms.evalLowthreshold(inputs);
	double hithresh = myParms.evalHighthreshold(inputs);
	bool l2grad = myParms.evalL2gradient(inputs);

	const uint32_t inheight = top->textureDesc.height;
	const uint32_t inwidth = top->textureDesc.width;
	

	TD::OP_CUDAAcquireInfo acquireInfo;

	acquireInfo.stream = myStream;
	const TD::OP_CUDAArrayInfo* inputArray = top->getCUDAArray(acquireInfo, nullptr);

	TD::TOP_CUDAOutputInfo info;
	info.textureDesc.width = top->textureDesc.width;
	info.textureDesc.height = top->textureDesc.height;
	info.textureDesc.texDim = top->textureDesc.texDim;
	info.textureDesc.pixelFormat = TD::OP_PixelFormat::Mono8Fixed;
	info.stream = myStream;

	const TD::OP_CUDAArrayInfo* outputInfo = output->createCUDAArray(info, nullptr);
	if (!outputInfo)
		return;

	// Now that we have gotten all of the pointers to the OP_CUDAArrayInfos that we may want, we can tell the context
	// that we are going to start doing CUDA operations. This will cause the cudaArray members of the OP_CUDAArrayInfo
	// to get filled in with valid addresses.
	if (!myContext->beginCUDAOperations(nullptr))
		return;
	
	if (inputArray->cudaArray == nullptr)
	{
		myError = "CUDA memory for input TOP was not mapped correctly.";
		return;
	}

	*myFrame = cv::cuda::GpuMat(inheight, inwidth, myMatType);
	GpuUtils::arrayToMatGPU(inwidth, inheight, inputArray->cudaArray, *myFrame, myPixelSize);

	if (myFrame->empty())
		return;

	if (appertureSize % 2 == 0)
		++appertureSize;

	int kernel = appertureSize;
	if (kernel > 31)			// clamp at 31
		kernel = 31;

	// converting CV_16F - and other formats - to CV_8U will throw OpenCV error - unsupported
	// https://github.com/opencv/opencv/issues/23101
	myFrame->convertTo(*myFrame, CV_8U);
	if (myNumChan > 1)
	{
		GpuMat chan[4];
		split(*myFrame, chan);
		*myFrame = chan[0];
	}


	auto cannyEdge = createCannyEdgeDetector(255 * lothresh,
											 255 * hithresh,
											 kernel,
											 l2grad ? true : false);

	cv::cuda::Stream stream = cv::cuda::StreamAccessor::wrapStream(myStream);
	cannyEdge->detect(*myFrame, *myFrame, stream);

	GpuUtils::matGPUToArray(info.textureDesc.width, info.textureDesc.height, *myFrame, outputInfo->cudaArray, 1);

	myContext->endCUDAOperations(nullptr);
}

void
CannyEdgeTOP::setupParameters(TD::OP_ParameterManager* manager, void*)
{
	myParms.setup(manager);
}

void 
CannyEdgeTOP::getErrorString(TD::OP_String* error, void*)
{
	error->setString(myError.c_str());
	myError.clear();
}


bool
CannyEdgeTOP::checkInputTop(const TD::OP_TOPInput* topInput)
{
	switch (topInput->textureDesc.pixelFormat)
	{
	case TD::OP_PixelFormat::A8Fixed:
	case TD::OP_PixelFormat::A16Fixed:
	case TD::OP_PixelFormat::A16Float:
	case TD::OP_PixelFormat::A32Float:
	case TD::OP_PixelFormat::Mono8Fixed:
	case TD::OP_PixelFormat::Mono16Fixed:
	case TD::OP_PixelFormat::Mono16Float:
	case TD::OP_PixelFormat::Mono32Float:
		myNumChan = 1;
		break;
	case TD::OP_PixelFormat::MonoA8Fixed:
	case TD::OP_PixelFormat::MonoA16Fixed:
	case TD::OP_PixelFormat::MonoA16Float:
	case TD::OP_PixelFormat::MonoA32Float:
	case TD::OP_PixelFormat::RG8Fixed:
	case TD::OP_PixelFormat::RG16Fixed:
	case TD::OP_PixelFormat::RG16Float:
	case TD::OP_PixelFormat::RG32Float:
		myNumChan = 2;
		break;
		// RGB has alpha on its channels
	case TD::OP_PixelFormat::RGBA8Fixed:
	case TD::OP_PixelFormat::RGBA16Fixed:
	case TD::OP_PixelFormat::RGBA16Float:
	case TD::OP_PixelFormat::RGBA32Float:
		myNumChan = 4;
		break;
	default:
		myError = "Pixel format not supported.";
		myNumChan = -1;
		return false;
	}

	switch (topInput->textureDesc.pixelFormat)
	{
	case TD::OP_PixelFormat::A8Fixed:
	case TD::OP_PixelFormat::Mono8Fixed:
	case TD::OP_PixelFormat::MonoA8Fixed:
	case TD::OP_PixelFormat::RG8Fixed:
	case TD::OP_PixelFormat::RGBA8Fixed:
		myMatType = CV_MAKETYPE(CV_8U, myNumChan);
		myPixelSize = 1 * myNumChan;
		break;
	case TD::OP_PixelFormat::A16Float:
	case TD::OP_PixelFormat::Mono16Float:
	case TD::OP_PixelFormat::MonoA16Float:
	case TD::OP_PixelFormat::RG16Float:
	case TD::OP_PixelFormat::RGBA16Float:
		myMatType = CV_MAKETYPE(CV_16F, myNumChan);
		myPixelSize = 2 * myNumChan;
		break;
	case TD::OP_PixelFormat::A16Fixed:
	case TD::OP_PixelFormat::Mono16Fixed:
	case TD::OP_PixelFormat::MonoA16Fixed:
	case TD::OP_PixelFormat::RG16Fixed:
	case TD::OP_PixelFormat::RGBA16Fixed:
		myMatType = CV_MAKETYPE(CV_16U, myNumChan);
		myPixelSize = 2 * myNumChan;
		break;
	case TD::OP_PixelFormat::A32Float:
	case TD::OP_PixelFormat::Mono32Float:
	case TD::OP_PixelFormat::MonoA32Float:
	case TD::OP_PixelFormat::RG32Float:
	case TD::OP_PixelFormat::RGBA32Float:
		myMatType = CV_MAKETYPE(CV_32F, myNumChan);
		myPixelSize = 2 * myNumChan;
		break;
	}

	return true;
}
