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

#include "CudaTOP.h"
#include "Parameters.h"

#include <cassert>
#include <string>
#include <array>

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
	info->executeMode = TOP_ExecuteMode::CUDA;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Cuda");
	// English readable name
	customInfo.opLabel->setString("Cuda");
	// Information of the author of the node
	customInfo.authorName->setString("Gabriel Robels");
	customInfo.authorEmail->setString("support@derivative.ca");

	// This TOP takes zero to one input
	customInfo.minInputs = 0;
	customInfo.maxInputs = 1;
}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context*)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll
	return new CudaTOP(info);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (CudaTOP*)instance;
}

};

CudaTOP::CudaTOP(const OP_NodeInfo* info)
{
}

CudaTOP::~CudaTOP()
{
}

void
CudaTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	ginfo->cookEveryFrameIfAsked = false;
}

bool
CudaTOP::getOutputFormat(TOP_OutputFormat*, const OP_Inputs*, void*)
{
	return false;
}

extern void doCUDAOperation(int width, int height, cudaArray* input, cudaArray* output, std::array<uint8_t, 4> rgba);

void
CudaTOP::execute(TOP_OutputFormatSpecs* output, const OP_Inputs* inputs, TOP_Context*, void*)
{
	checkOutputFormat(output);

	cudaArray* data = nullptr;
	if (inputs->getNumInputs() > 0)
	{
		const OP_TOPInput* top = inputs->getInputTOP(0);
		if (top)
		{
			checkTopFormat(top, output);
			data = top->cudaInput;
		}
	}

	if (inputs->getNumInputs() == 0) {
		Color myColor = myParms.evalColor(inputs);
		myRgba8.at(0) = static_cast<uint8_t>(UINT8_MAX * myColor.r);
		myRgba8.at(1) = static_cast<uint8_t>(UINT8_MAX * myColor.g);
		myRgba8.at(2) = static_cast<uint8_t>(UINT8_MAX * myColor.b);
		myRgba8.at(3) = static_cast<uint8_t>(UINT8_MAX * myColor.a);
	}

	doCUDAOperation(output->width, output->height, data, output->cudaOutput[0], myRgba8);
}

void 
CudaTOP::setupParameters(OP_ParameterManager* manager, void*)
{
	myParms.setup(manager);
}

void 
CudaTOP::getErrorString(OP_String* error, void*)
{
	error->setString(myError.c_str());
	myError.clear();
}

void 
CudaTOP::checkOutputFormat(const TOP_OutputFormatSpecs* outputFormat)
{
	if (outputFormat->redBits != 8 ||
		outputFormat->greenBits != 8 ||
		outputFormat->blueBits != 8 ||
		outputFormat->alphaBits != 8)
	{
		myError = "CUDA Kernel is currently only written to handle 8-bit RGBA textures.";
		return;
	}
}

void 
CudaTOP::checkTopFormat(const OP_TOPInput* topInput, const TOP_OutputFormatSpecs* outputFormat)
{
	if (topInput->width != outputFormat->width ||
		topInput->height != outputFormat->height)
	{
		myError = "Input and output resolution must be the same.";
		return;
	}
	if (topInput->pixelFormat != GL_RGBA8)
	{
		myError = "CUDA Kernel is currently only written to handle 8-bit RGBA textures.";
		return;
	}
	if (topInput->cudaInput == nullptr)
	{
		myError = "CUDA memory for input TOP was not mapped correctly.";
	}
}
