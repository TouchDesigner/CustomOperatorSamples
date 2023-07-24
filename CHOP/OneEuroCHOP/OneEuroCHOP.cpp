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

#include "OneEuroCHOP.h"
#include "OneEuroImpl.h"

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
	customInfo.opType->setString("Oneeuro");
	// English readable name
	customInfo.opLabel->setString("One Euro");
	// Information of the author of the node
	customInfo.authorName->setString("Author Name");
	customInfo.authorEmail->setString("email@email");

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
	return new OneEuroCHOP(info);
}

DLLEXPORT
void
DestroyCHOPInstance(CHOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (OneEuroCHOP*)instance;
}

};


OneEuroCHOP::OneEuroCHOP(const OP_NodeInfo*) :
	myFiltersPerChannel{}
{

}

OneEuroCHOP::~OneEuroCHOP()
{
	for (OneEuroImpl* filter : myFiltersPerChannel)
	{
		delete filter;
	}
}

void
OneEuroCHOP::getGeneralInfo(CHOP_GeneralInfo* ginfo, const OP_Inputs* input, void*)
{
	// This will cause the node to cook every frame if the output is used
	ginfo->cookEveryFrameIfAsked = true;

	// The output matches the sample and channels of the input with index 0
	ginfo->inputMatchIndex = 0;

	// We want the output to be time sliced
	ginfo->timeslice = true;
}

bool
OneEuroCHOP::getOutputInfo(CHOP_OutputInfo* info, const OP_Inputs* inputs, void*)
{
	// We return false since the signal matches the input 0
	return false;
}

void
OneEuroCHOP::getChannelName(int32_t index, OP_String *name, const OP_Inputs*, void*)
{
	name->setString("chan1");
}

void
OneEuroCHOP::execute(CHOP_Output* output,
							  const OP_Inputs* inputs,
							  void*)
{
	const OP_CHOPInput* chop = inputs->getInputCHOP(0);
	if (!chop)
		return;

	handleParameters(inputs, chop);

	int inputSampleIdx = 0;
	for (int i = 0; i < output->numChannels; ++i)
	{
		for (int j = 0; j < output->numSamples; ++j)
		{
			inputSampleIdx = (inputSampleIdx + 1) % chop->numSamples;
			output->channels[i][j] = (float)myFiltersPerChannel.at(i)->filter(chop->channelData[i][inputSampleIdx]);
		}
	}
}

void
OneEuroCHOP::setupParameters(OP_ParameterManager* manager, void*)
{
	{
		OP_NumericParameter p;
		p.name = "Mincutoff";
		p.label = "Cutoff Frequency(Hz)";
		p.page = "Filter";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = 0.0;
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
		p.name = "Beta";
		p.label = "Speed Coefficient";
		p.page = "Filter";
		p.defaultValues[0] = 0.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Dcutoff";
		p.label = "Slope Cutoff Frequency (Hz)";
		p.page = "Filter";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}
}

void
OneEuroCHOP::handleParameters(const OP_Inputs* input, const OP_CHOPInput* chop)
{
	double	minCutOff = input->getParDouble("Mincutoff");
	double	beta = input->getParDouble("Beta");
	double	dCutOff = input->getParDouble("Dcutoff");
	double	rate = chop->sampleRate;
	int		numChannels = chop->numChannels;

	// Update existing filters
	for (int i = 0; i < myFiltersPerChannel.size(); ++i)
	{
		myFiltersPerChannel.at(i)->changeInput(rate, minCutOff, beta, dCutOff);
	}

	// Create new filters if needed
	for (int i = (int)myFiltersPerChannel.size(); i < numChannels; ++i)
	{
		myFiltersPerChannel.emplace_back(new OneEuroImpl(rate, minCutOff, beta, dCutOff));
	}
}
