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
#include "GpuUtils.cuh"

#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "cuda_runtime.h"



#pragma region Menus
enum class ModeMenuItems
{
	dft,
	idft
};

enum class CoordMenuItems
{
	polar,
	cartesian
};

enum class ChanMenuItems
{
	r,
	g,
	b,
	a
};

#pragma endregion

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
	customInfo.authorName->setString("Author Name");
	customInfo.authorEmail->setString("email@email");

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
	return new SpectrumTOP(info, context);
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


SpectrumTOP::SpectrumTOP(const OP_NodeInfo*, TOP_Context *context) :
	myFrame( new cv::cuda::GpuMat() ), 
	myResult( new cv::cuda::GpuMat() ), 
	myExecuteCount(0),
	myError(""),
	myNumChan(-1),
	myContext(context),
	myChanFormat(GpuUtils::ChannelFormat::U16)
{
	cudaStreamCreate(&myStream);
}

SpectrumTOP::~SpectrumTOP()
{
	delete myFrame;
	delete myResult;
	cudaStreamDestroy(myStream);
}

void
SpectrumTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	ginfo->cookEveryFrameIfAsked = false;
}

void
SpectrumTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
	using namespace cv::cuda;

	myError = "";
	myExecuteCount++;

	const OP_TOPInput* top = inputs->getInputTOP(0);
	if (!top || !checkInputTop(top, inputs))
		return;

	ModeMenuItems mode = static_cast<ModeMenuItems>(inputs->getParInt("Mode"));
	int channame = inputs->getParInt("Chan");
	bool transrows = inputs->getParInt("Transrows");
	CoordMenuItems coord = static_cast<CoordMenuItems>(inputs->getParInt("Coord"));

	mySize.width = top->textureDesc.width;
	mySize.height = top->textureDesc.height;

	OP_CUDAAcquireInfo acquireInfo;

	acquireInfo.stream = myStream;
	const OP_CUDAArrayInfo* inputArray = top->getCUDAArray(acquireInfo, nullptr);


	TOP_CUDAOutputInfo info;
	info.textureDesc.width = top->textureDesc.width;
	info.textureDesc.height = top->textureDesc.height;
	info.textureDesc.texDim = top->textureDesc.texDim;
	info.textureDesc.pixelFormat = (mode == ModeMenuItems::dft) ? OP_PixelFormat::RG32Float : OP_PixelFormat::Mono32Float;
	info.stream = myStream;

	const OP_CUDAArrayInfo* outputInfo = output->createCUDAArray(info, nullptr);
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

	*myFrame = cv::cuda::GpuMat(info.textureDesc.height, info.textureDesc.width, CV_32FC2);
	if (mode == ModeMenuItems::dft)
	{
		GpuUtils::arrayToComplexMatGPU(info.textureDesc.width, info.textureDesc.height, inputArray->cudaArray, *myFrame, myNumChan, channame, myChanFormat);
	}
	else
	{
		GpuUtils::arrayToMatGPU(info.textureDesc.width, info.textureDesc.height, inputArray->cudaArray, *myFrame, 2 * sizeof(float));
	}

	if (myFrame->empty())
		return;
	

	if (mode == ModeMenuItems::dft)
	{
		dft(*myFrame, *myResult, mySize, 0);

		if (!transrows)
		{
			swapQuadrants(*myResult);
		}
		else
		{
			swapSides(*myResult);
		}

		if (coord == CoordMenuItems::polar)
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
		if (coord == CoordMenuItems::polar)
		{
			GpuMat channels[2];
			split(*myFrame, channels);

			cv::cuda::exp(channels[0], channels[0]);
			add(channels[0], -1, channels[0]);
			polarToCart(channels[0], channels[1], channels[0], channels[1]);

			merge(channels, 2, *myFrame);
		}

		if (!transrows)
		{
			swapQuadrants(*myFrame);
		}
		else
		{
			swapSides(*myFrame);
		}

		dft(*myFrame, *myResult, mySize, 0);
	}

	

	if (mode == ModeMenuItems::dft)
	{
		GpuUtils::matGPUToArray(info.textureDesc.width, info.textureDesc.height, *myResult, outputInfo->cudaArray, 2 * sizeof(float));
	}
	else
	{
		GpuUtils::complexMatGPUToArray(info.textureDesc.width, info.textureDesc.height, *myResult, outputInfo->cudaArray);
	}

	myContext->endCUDAOperations(nullptr);
}

void
SpectrumTOP::setupParameters(OP_ParameterManager* manager, void*)
{
	{
		OP_StringParameter p;
		p.name = "Mode";
		p.label = "Mode";
		p.page = "Spectrum";
		p.defaultValue = "dft";
		std::array<const char*, 2> Names =
		{
			"dft",
			"idft"
		};
		std::array<const char*, 2> Labels =
		{
			"Discrete Fourier Transform",
			"Inverse Discrete Fourier Transform"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = "Coord";
		p.label = "Coordinate System";
		p.page = "Spectrum";
		p.defaultValue = "polar";
		std::array<const char*, 2> Names =
		{
			"polar",
			"cartesian"
		};
		std::array<const char*, 2> Labels =
		{
			"Polar (Magnitude, Phase)",
			"Cartesian (Real, Imaginary)"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = "Chan";
		p.label = "Channel";
		p.page = "Spectrum";
		p.defaultValue = "r";
		std::array<const char*, 4> Names =
		{
			"r",
			"g",
			"b",
			"a"
		};
		std::array<const char*, 4> Labels =
		{
			"R",
			"G",
			"B",
			"A"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Transrows";
		p.label = "Transform Rows";
		p.page = "Spectrum";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}
}

void 
SpectrumTOP::getErrorString(OP_String* error, void*)
{
	error->setString(myError.c_str());
	myError.clear();
}


bool
SpectrumTOP::checkInputTop(const OP_TOPInput* topInput, const OP_Inputs* input)
{
	ModeMenuItems myMode = static_cast<ModeMenuItems>(input->getParInt("Mode"));
	if (myMode == ModeMenuItems::idft && topInput->textureDesc.pixelFormat != OP_PixelFormat::RG32Float)
	{
		myError = "Inverse transform requires a 32-bit float RG texture.";
		return false;
	}

	ChanMenuItems myChan = static_cast<ChanMenuItems>(input->getParInt("Chan"));
	switch (topInput->textureDesc.pixelFormat)
	{
		case OP_PixelFormat::A8Fixed:
		case OP_PixelFormat::A16Fixed:
		case OP_PixelFormat::A16Float:
		case OP_PixelFormat::A32Float:
			// Only A channel is valid, change to use channel as index
			if (myChan == ChanMenuItems::a)
				myChan = ChanMenuItems::r;
			else
				myChan = ChanMenuItems::r; // ::Invalid what is invalid here ?
		case OP_PixelFormat::Mono8Fixed:
		case OP_PixelFormat::Mono16Fixed:
		case OP_PixelFormat::Mono16Float:
		case OP_PixelFormat::Mono32Float:
			myNumChan = 1;
			break;
		case OP_PixelFormat::MonoA8Fixed:
		case OP_PixelFormat::MonoA16Fixed:
		case OP_PixelFormat::MonoA16Float:
		case OP_PixelFormat::MonoA32Float:
			// Only RA channels are valid, change to use channel as index
			if (myChan == ChanMenuItems::a)
				myChan = ChanMenuItems::r; // ::Second what is Second here ?
			else if (myChan != ChanMenuItems::r)
				myChan = ChanMenuItems::r;  // ::Invalid what is Invalid here ?
		case OP_PixelFormat::RG8Fixed:
		case OP_PixelFormat::RG16Fixed:
		case OP_PixelFormat::RG16Float:
		case OP_PixelFormat::RG32Float:
			myNumChan = 2;
			break;
		// RGB has alpha on its channels
		case OP_PixelFormat::RGBX16Float:
		case OP_PixelFormat::RGBX32Float:
			if (myChan == ChanMenuItems::a)
				myChan = ChanMenuItems::r; // ::Invalid what is Invalid here ?
		case OP_PixelFormat::BGRA8Fixed:
		case OP_PixelFormat::RGBA8Fixed:
		case OP_PixelFormat::RGBA16Fixed:
		case OP_PixelFormat::RGBA16Float:
		case OP_PixelFormat::RGBA32Float:
			myNumChan = 4;
			break;
		default:
			myError = "Pixel format not supported.";
			myNumChan = -1;
			return false;
	}

	switch (topInput->textureDesc.pixelFormat)
	{
		case OP_PixelFormat::A8Fixed:
		case OP_PixelFormat::Mono8Fixed:
		case OP_PixelFormat::MonoA8Fixed:
		case OP_PixelFormat::RG8Fixed:
		case OP_PixelFormat::RGBA8Fixed:
		case OP_PixelFormat::BGRA8Fixed:
			myChanFormat = GpuUtils::ChannelFormat::U8;
			break;
		case OP_PixelFormat::A16Float:
		case OP_PixelFormat::Mono16Float:
		case OP_PixelFormat::MonoA16Float:
		case OP_PixelFormat::RG16Float:
		case OP_PixelFormat::RGBX16Float:
		case OP_PixelFormat::RGBA16Float:
			myChanFormat = GpuUtils::ChannelFormat::F16;
			break;
		case OP_PixelFormat::A16Fixed:
		case OP_PixelFormat::Mono16Fixed:
		case OP_PixelFormat::MonoA16Fixed:
		case OP_PixelFormat::RG16Fixed:
		case OP_PixelFormat::RGBA16Fixed:
			myChanFormat = GpuUtils::ChannelFormat::U16;
			break;
		case OP_PixelFormat::A32Float:
		case OP_PixelFormat:: Mono32Float:
		case OP_PixelFormat::MonoA32Float:
		case OP_PixelFormat::RG32Float:
		case OP_PixelFormat::RGBX32Float:
		case OP_PixelFormat::RGBA32Float:
			myChanFormat = GpuUtils::ChannelFormat::F32;
			break;
	}

	if (myMode == ModeMenuItems::dft && static_cast<int>(myChan) >= myNumChan)
	{
		myError = "Channel not available.";
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
