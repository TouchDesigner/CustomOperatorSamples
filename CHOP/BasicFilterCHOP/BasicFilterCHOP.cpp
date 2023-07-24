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

#include "BasicFilterCHOP.h"

#include <cassert>
#include <string>

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

DLLEXPORT
void
FillCHOPPluginInfo(CHOP_PluginInfo *info)
{
	// For more information on CHOP_PluginInfo see CHOP_CPlusPlusBase.h

	// Always set this to CHOPCPlusPlusAPIVersion.
	info->apiVersion = CHOPCPlusPlusAPIVersion;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Basicfilter");
	// English readable name
	customInfo.opLabel->setString("Basic Filter");
	// Information of the author of the node
	customInfo.authorName->setString("Author Name");
	customInfo.authorEmail->setString("email@email.ca");

	// This CHOP takes one input
	customInfo.minInputs = 1;
	customInfo.maxInputs = 1;
}

DLLEXPORT
CHOP_CPlusPlusBase*
CreateCHOPInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per CHOP that is using the .dll
	return new BasicFilterCHOP(info);
}

DLLEXPORT
void
DestroyCHOPInstance(CHOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (BasicFilterCHOP*)instance;
}

};


BasicFilterCHOP::BasicFilterCHOP(const OP_NodeInfo*)
{

}

BasicFilterCHOP::~BasicFilterCHOP()
{

}

void
BasicFilterCHOP::getGeneralInfo(CHOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	// This will cause the node to cook every frame if the output is used
	ginfo->cookEveryFrameIfAsked = false;

	// The output matches the sample and channels of the input with index 0
	ginfo->inputMatchIndex = 0;

	ginfo->timeslice = false;
}

bool
BasicFilterCHOP::getOutputInfo(CHOP_OutputInfo* info, const OP_Inputs* inputs, void*)
{
	// We return false since the signal matches the input 0
	return false;
}

void
BasicFilterCHOP::getChannelName(int32_t index, OP_String *name, const OP_Inputs*, void*)
{
	name->setString("chan1");
}

void
BasicFilterCHOP::execute(CHOP_Output* output,
							  const OP_Inputs* inputs,
							  void*)
{
	// Get all Parameters
	bool	applyScale = inputs->getParInt("Applyscale") ? true : false;
	double	scale = inputs->getParDouble("Scale");
	bool	applyOffset = inputs->getParInt("Applyoffset") ? true : false;
	double	offset = inputs->getParDouble("Offset");

	inputs->enablePar("Scale", applyScale);
	inputs->enablePar("Offset", applyOffset);

	const OP_CHOPInput* input = inputs->getInputCHOP(0);

	if (!input)
		return;

	// We know input and output have the same numChannels since we returned 
	// false in getOutputInfo and it is not timeSliced
	for (int i = 0; i < output->numChannels; ++i)
	{
		for (int j = 0; j < output->numSamples; ++j)
		{
			output->channels[i][j] = static_cast<float>(input->channelData[i][j] * scale + offset);
		}
	}
}

void
BasicFilterCHOP::setupParameters(OP_ParameterManager* manager, void*)
{
	{
		OP_NumericParameter p;
		p.name = "Applyscale";
		p.label = "Apply Scale";
		p.page = "Filter";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Scale";
		p.label = "Scale";
		p.page = "Filter";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = -10.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Applyoffset";
		p.label = "Apply Offset";
		p.page = "Filter";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Offset";
		p.label = "Offset";
		p.page = "Filter";
		p.defaultValues[0] = 0.0;
		p.minSliders[0] = -10.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}
}
