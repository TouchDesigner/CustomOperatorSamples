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

#include "BasicGeneratorCHOP.h"

#include <string>
#include <array>
#include <cassert>
#include <cmath>

enum class OperationMenuItems
{
	Add,
	Multiply,
	Power
};

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
	customInfo.opType->setString("Basicgenerator");
	// English readable name
	customInfo.opLabel->setString("Basic Generator");
	// Information of the author of the node
	customInfo.authorName->setString("Author Name");
	customInfo.authorEmail->setString("email@email");

	// This CHOP takes no inputs
	customInfo.minInputs = 0;
	customInfo.maxInputs = 0;
}

DLLEXPORT
CHOP_CPlusPlusBase*
CreateCHOPInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per CHOP that is using the .dll
	return new BasicGeneratorCHOP(info);
}

DLLEXPORT
void
DestroyCHOPInstance(CHOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (BasicGeneratorCHOP*)instance;
}

};


BasicGeneratorCHOP::BasicGeneratorCHOP(const OP_NodeInfo*)
{

}

BasicGeneratorCHOP::~BasicGeneratorCHOP()
{

}

void
BasicGeneratorCHOP::getGeneralInfo(CHOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	// This chop doesn't need to cook every frame
	ginfo->cookEveryFrameIfAsked = false;

	// Note: To disable timeslicing you'll need to turn this off, as well as ensure that
	// getOutputInfo() returns true, and likely also set the info->numSamples to how many
	// samples you want to generate for this CHOP. Otherwise it'll take on length of the
	// input CHOP, which may be timesliced.
	ginfo->timeslice = false;
}

bool
BasicGeneratorCHOP::getOutputInfo(CHOP_OutputInfo* info, const OP_Inputs* inputs, void*)
{
	info->numChannels = inputs->getParInt("Numberofchannels");
	info->numSamples = inputs->getParInt("Length");
	info->startIndex = 0;
	return true;
}

void
BasicGeneratorCHOP::getChannelName(int32_t index, OP_String *name, const OP_Inputs*, void*)
{
	std::string channelName = "chan" + std::to_string(index);
	name->setString(channelName.c_str());
}

void
BasicGeneratorCHOP::execute(CHOP_Output* output,
							  const OP_Inputs* inputs,
							  void*)
{
	// Get all Parameters
	bool	applyScale = inputs->getParInt("Applyscale") ? true : false;
	double	scale = inputs->getParDouble("Scale");
	OperationMenuItems		operation = static_cast<OperationMenuItems>(inputs->getParInt("Operation"));

	inputs->enablePar("Scale", applyScale);

	// Get length and channels from the output since they first need to be set to the current size
	int		length = output->numSamples;
	int		channels = output->numChannels;

	// Calculate scale*(channel operation sample)
	double	curValue;
	for (int i = 0; i < channels; ++i) 
	{
		for (int j = 0; j < length; ++j) 
		{
			switch (operation) 
			{
				case OperationMenuItems::Add:
				{
					curValue = (double)i + j;
					break;
				}
				case OperationMenuItems::Multiply:
				{
					curValue = (double)i * j;
					break;
				}
				case OperationMenuItems::Power:
				default:
				{
					curValue = std::pow((double)i, (double)j);
					break;
				}
			}

			curValue *= scale;

			output->channels[i][j] = (float)curValue;
		}
	}
}

void
BasicGeneratorCHOP::setupParameters(OP_ParameterManager* manager, void*)
{
	{
		OP_NumericParameter p;
		p.name = "Length";
		p.label = "Length";
		p.page = "Generator Basic";
		p.defaultValues[0] = 10;
		p.minSliders[0] = 1.0;
		p.maxSliders[0] = 50.0;
		p.minValues[0] = 1.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Numberofchannels";
		p.label = "Number Of Channels";
		p.page = "Generator Basic";
		p.defaultValues[0] = 10;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 50.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Applyscale";
		p.label = "Apply Scale";
		p.page = "Generator Basic";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Scale";
		p.label = "Scale";
		p.page = "Generator Basic";
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
		OP_StringParameter p;
		p.name = "Operation";
		p.label = "Operation";
		p.page = "Generator Basic";
		p.defaultValue = "Add";
		std::array<const char*, 3> Names =
		{
			"Add",
			"Multiply",
			"Power"
		};
		std::array<const char*, 3> Labels =
		{
			"Add",
			"Multiply",
			"Power"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}
}
