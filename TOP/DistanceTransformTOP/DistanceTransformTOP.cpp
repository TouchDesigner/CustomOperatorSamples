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

#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


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
FillTOPPluginInfo(TD::TOP_PluginInfo* info)
{
	// This must always be set to this constant
	info->apiVersion = TD::TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TD::TOP_ExecuteMode::CPUMem;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	TD::OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Distancetransfrom");
	// English readable name
	customInfo.opLabel->setString("Distance Transform");
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
	return new DistanceTransformTOP(info, context);
}

DLLEXPORT
void
DestroyTOPInstance(TD::TOP_CPlusPlusBase* instance, TD::TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (DistanceTransformTOP*)instance;
}

};


DistanceTransformTOP::DistanceTransformTOP(const TD::OP_NodeInfo*, TD::TOP_Context *context) :
	myFrame{ new cv::Mat() },
	myExecuteCount{0},
	myPrevDownRes{nullptr},
	myContext{context}
{
}

DistanceTransformTOP::~DistanceTransformTOP()
{
	delete myFrame;
}

void
DistanceTransformTOP::getGeneralInfo(TD::TOP_GeneralInfo* ginfo, const TD::OP_Inputs*, void*)
{
	ginfo->cookEveryFrame = false;
	ginfo->cookEveryFrameIfAsked = false;
}

void
DistanceTransformTOP::execute(TD::TOP_Output* output, const TD::OP_Inputs* inputs, void*)
{
	using namespace cv;
	
	inputTopToMat(inputs);
	if (myFrame->empty())
		return;

	if (!myPrevDownRes)
		return;

	TD::TOP_UploadInfo info;
	info.textureDesc.width = myPrevDownRes->textureDesc.width;
	info.textureDesc.height = myPrevDownRes->textureDesc.height;
	info.textureDesc.texDim = TD::OP_TexDim::e2D;
	info.textureDesc.pixelFormat = TD::OP_PixelFormat::Mono32Float;
	info.colorBufferIndex = 0;

	int distanceType = getType(static_cast<DistancetypeMenuItems>(inputs->getParInt("Distancetype")));
	int maskSize = getMask(static_cast<MasksizeMenuItems>(inputs->getParInt("Masksize")));

	distanceTransform(*myFrame, *myFrame, distanceType, maskSize);
	
	bool donormalize = inputs->getParInt("Normalize") ? true : false;

	if (donormalize)
		normalize(*myFrame, *myFrame, 0, 1.0, NORM_MINMAX);

	cvMatToOutput(output, info);
}

void
DistanceTransformTOP::setupParameters(TD::OP_ParameterManager* manager, void*)
{
	{
		TD::OP_StringParameter p;
		p.name = "Distancetype";
		p.label = "Distance Type";
		p.page = "Transform";
		p.defaultValue = "L1";
		std::array<const char*, 3> Names =
		{
			"L1",
			"L2",
			"C"
		};
		std::array<const char*, 3> Labels =
		{
			"L1",
			"L2",
			"C"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = "Masksize";
		p.label = "Mask Size";
		p.page = "Transform";
		p.defaultValue = "Three";
		std::array<const char*, 3> Names =
		{
			"Three",
			"Five",
			"Precise"
		};
		std::array<const char*, 3> Labels =
		{
			"3x3",
			"5x5",
			"Precise"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = "Normalize";
		p.label = "Normalize";
		p.page = "Transform";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = "Channel";
		p.label = "Channel";
		p.page = "Transform";
		p.defaultValue = "R";
		std::array<const char*, 4> Names =
		{
			"R",
			"G",
			"B",
			"A"
		};
		std::array<const char*, 4> Labels =
		{
			"R",
			"G",
			"B",
			"A"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}
}

void
DistanceTransformTOP::cvMatToOutput(TD::TOP_Output* out, TD::TOP_UploadInfo info) const
{
	size_t	height = info.textureDesc.height;
	size_t	width = info.textureDesc.width;
	size_t imgsize = 1 * height * width * sizeof(float);

	TD::OP_SmartRef<TD::TOP_Buffer> buf = myContext->createOutputBuffer(imgsize, TD::TOP_BufferFlags::None, nullptr);
	float* pixel = static_cast<float*>(buf->data);

	cv::resize(*myFrame, *myFrame, cv::Size(width, height));
	cv::flip(*myFrame, *myFrame, 0);
	float* data = static_cast<float*>(static_cast<void*>(myFrame->data));

	memcpy(pixel, data, imgsize);
	out->uploadBuffer(&buf, info, nullptr);
}

void 
DistanceTransformTOP::inputTopToMat(const TD::OP_Inputs* in)
{
	const TD::OP_TOPInput*	top = in->getInputTOP(0);
	if (!top)
	{
		*myFrame = cv::Mat();
		return;
	}

	TD::OP_TOPInputDownloadOptions	opts;
	opts.verticalFlip = true;
	opts.pixelFormat = TD::OP_PixelFormat::RGBA8Fixed;

	TD::OP_SmartRef<TD::OP_TOPDownloadResult> downRes = top->downloadTexture(opts, nullptr);

	if (!downRes)
	{
		*myFrame = cv::Mat();
		return;
	}

	if (myPrevDownRes)
	{
		uint8_t* pixel = (uint8_t*)myPrevDownRes->getData();

		if (!pixel)
		{
			*myFrame = cv::Mat();
			return;
		}

		int	height = top->textureDesc.height;
		int	width = top->textureDesc.width;

		*myFrame = cv::Mat(height, width, CV_8UC1);
		uint8_t* data = (uint8_t*)myFrame->data;
		for (int i = 0; i < height; i += 1) {
			for (int j = 0; j < width; j += 1) {
				int pixelN = i * width + j;
				int index = 4 * pixelN + (in->getParInt("Channel"));
				data[pixelN] = pixel[index];
			}
		}
	}
	myPrevDownRes = std::move(downRes);
	
}
