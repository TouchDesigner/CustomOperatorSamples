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

#include "BasicFilterTOP.h"

#include "ThreadManager.h"
#include "FilterWork.h"
#include "Parameters.h"

#include <cassert>
#include <vector>
#include <cstdlib>

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
	customInfo.opType->setString("Filter");
	// English readable name
	customInfo.opLabel->setString("Filter");
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
	return new BasicFilterTOP(info);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (BasicFilterTOP*)instance;
}

};

BasicFilterTOP::BasicFilterTOP(const OP_NodeInfo* info) :
	myThreadManagers{}, myExecuteCount{ 0 }, myMultiThreaded{ false }, myParms{ new Parameters() }
{
}

BasicFilterTOP::~BasicFilterTOP()
{
	delete myParms;
}

void
BasicFilterTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	ginfo->cookEveryFrameIfAsked = false;
    ginfo->memPixelType = OP_CPUMemPixelType::BGRA8Fixed;
	ginfo->inputSizeIndex = 0;
}

bool
BasicFilterTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs*, void*)
{
	return false;
}


void
BasicFilterTOP::execute(TOP_OutputFormatSpecs* output, const OP_Inputs* inputs, TOP_Context *context, void*)
{
	const OP_TOPInput*	top = inputs->getInputTOP(0);

	if (!top)
		return;

	int inHeight = top->height;
	int inWidth = top->width;

	OP_TOPInputDownloadOptions	opts;
	opts.cpuMemPixelType = OP_CPUMemPixelType::BGRA8Fixed;
	switch (myParms->evalDownloadtype(inputs))
	{
	case DownloadtypeMenuItems::Delayed:
		opts.downloadType = OP_TOPInputDownloadType::Delayed;
		break;
	case DownloadtypeMenuItems::Instant:
		opts.downloadType = OP_TOPInputDownloadType::Instant;
		break;
	}
	uint32_t*	inBuffer = static_cast<uint32_t*>(inputs->getTOPDataInCPUMemory(top, &opts));

	if (!inBuffer)
		return;

	if (myMultiThreaded)
	{
		for (ThreadManager* tm : myThreadManagers)
		{
			tm->syncParms(*myParms, inWidth, inHeight, output->width, output->height, inputs);
		}

		int threadId = std::abs(myExecuteCount % 3);
		myThreadManagers.at(threadId)->syncBuffer(inBuffer, static_cast<uint32_t*>(output->cpuPixelData[threadId]));
		myExecuteCount++;
		output->newCPUPixelDataLocation = myExecuteCount >= 0 ? myExecuteCount % 3 : -1;
	}
	else
	{
		Filter::doFilterWork(
			inBuffer, inWidth, inHeight, static_cast<uint32_t*>(output->cpuPixelData[0]), output->width, 
			output->height, myParms->evalDither(inputs), myParms->evalBitspercolor(inputs)
		);
		output->newCPUPixelDataLocation = 0;
	}

	bool threaded = myParms->evalMultithreaded(inputs);

	if (threaded & !myMultiThreaded)
		switchToMultiThreaded();

	if (!threaded & myMultiThreaded)
		switchToSingleThreaded();
}

void 
BasicFilterTOP::setupParameters(OP_ParameterManager* manager, void*)
{
	myParms->setup(manager);
}

void 
BasicFilterTOP::switchToSingleThreaded()
{
	for (int i = 0; i < NumCPUPixelDatas; ++i)
	{
		delete myThreadManagers.at(i);
		myThreadManagers.at(i) = nullptr;
	}

	myMultiThreaded = false;
}

void 
BasicFilterTOP::switchToMultiThreaded()
{
	// This delays the output by 3 frames
	myExecuteCount = -3;

	for (int i = 0; i < NumCPUPixelDatas; ++i)
	{
		myThreadManagers.at(i) = new ThreadManager();
	}

	myMultiThreaded = true;
}
